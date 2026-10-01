"""Edge benchmark: what the Pi actually delivers (plan §6 targets), measured, not assumed.

  python -m edge.bench --images DIR [--bundle-dir D | --model mri] [--n 50] [--deep 3] [--llm 3] [--out bench.json]

Measures, after a warm-up:
  analyze      end-to-end engine.analyze() per image (decode, preprocess, INT8 ONNX, exact CAM, Finer-CAM,
               conformal, OOD, overlays): median / p95 wall time and the engine's per-stage timings
  finer        re-rendering Finer-CAM at a new gamma (the slider)
  deep         Deep Explain (FP32 LayerCAM), if the bundle and PyTorch allow it
  llm          one assistant narrative: tokens/s for prompt and generation from llama-server's own timings
  system       peak RSS, SoC temperature before/after, throttling flags (vcgencmd), CPU model
Targets (plan §6): classifier + seg INT8 < 400 ms, exact CAM + Finer-CAM < 5 ms, Deep Explain < 6 s.
Run on the Pi with the active cooler fitted; note the ambient temperature.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".dcm", ".dicom", ".bmp", ".tif", ".tiff"}


def soc_temp_c() -> float | None:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except (OSError, ValueError):
        return None


THROTTLE_BITS = {0: "under-voltage now", 1: "ARM clock capped now", 2: "throttled now", 3: "soft temp limit now",
                 16: "under-voltage occurred", 17: "ARM clock capping occurred", 18: "throttling occurred",
                 19: "soft temp limit occurred"}


def throttled() -> dict | str | None:
    """vcgencmd get_throttled, decoded (0x0 = never throttled / under-volted since boot).
    Needs access to /dev/vcio_gencmd (group "video"); otherwise reported as unavailable, not as an error string."""
    if not shutil.which("vcgencmd"):
        return None
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out.startswith("throttled="):
        return "unavailable (no access to /dev/vcio_gencmd; run `vcgencmd get_throttled` as a video-group user)"
    v = int(out.split("=", 1)[1], 16)
    return {"raw": hex(v), "flags": [t for b, t in THROTTLE_BITS.items() if v >> b & 1]}


def peak_rss_mb() -> float | None:
    try:
        import resource                      # Linux / macOS (the Pi)
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / 1024 if sys.platform != "darwin" else r / 2**20, 1)
    except ImportError:
        try:
            import psutil                    # Windows dev box, if available
            return round(psutil.Process().memory_info().peak_wset / 2**20, 1)
        except (ImportError, AttributeError):
            return None


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith(("Model", "model name")):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def stats_ms(xs: list[float]) -> dict:
    a = np.asarray(xs)
    return {"n": int(a.size), "median": round(float(np.median(a)), 1), "p95": round(float(np.percentile(a, 95)), 1),
            "min": round(float(a.min()), 1), "max": round(float(a.max()), 1)}


def llm_bench(result: dict, url: str, runs: int) -> dict:
    from edge.agent.orchestrator import SYSTEM, Assistant
    from edge.agent.tools import case_facts

    a = Assistant()
    if not a.llm.available():
        return {"available": False}
    facts = case_facts(result)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": a._context(facts, a.kb.search(f"{facts['prediction']} MRI appearance", k=2))
             + "\n\nWrite a short narrative summary of this model output for the radiologist."}]
    out = []
    for _ in range(runs):
        body = json.dumps({"messages": msgs, "max_tokens": 260, "temperature": 0.2,
                           "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}).encode()
        t0 = time.perf_counter()
        req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            j = json.loads(r.read())
        t = j.get("timings", {})
        out.append({"wall_s": round(time.perf_counter() - t0, 2), "prompt_tokens": t.get("prompt_n"),
                    "prompt_tok_s": round(t.get("prompt_per_second", 0), 1), "gen_tokens": t.get("predicted_n"),
                    "gen_tok_s": round(t.get("predicted_per_second", 0), 1)})
    return {"available": True, "runs": out,
            "median_wall_s": statistics.median(o["wall_s"] for o in out),
            "median_gen_tok_s": statistics.median(o["gen_tok_s"] for o in out)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--bundle-dir", help="plaintext bundle directory (development)")
    src.add_argument("--model", default="mri", help="enrolled model name (decrypted + verified in RAM)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--deep", type=int, default=3, help="Deep Explain runs (0 = skip)")
    ap.add_argument("--llm", type=int, default=3, help="LLM narrative runs (0 = skip)")
    ap.add_argument("--llm-url", default="http://127.0.0.1:8081")
    ap.add_argument("--out", type=Path, default=Path("bench.json"))
    args = ap.parse_args()

    from edge.engine import Engine

    imgs = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in EXTS)
    if not imgs:
        sys.exit(f"no images in {args.images}")
    report = {"cpu": cpu_model(), "machine": platform.machine(), "python": platform.python_version(),
              "threads": args.threads, "temp_start_c": soc_temp_c(), "throttled_start": throttled()}

    t0 = time.perf_counter()
    if args.bundle_dir:
        eng = Engine.from_dir(args.bundle_dir, threads=args.threads)
    else:
        from edge.sync import Device
        files, version = Device().load_bundle(args.model)
        eng = Engine(files, threads=args.threads)
        report["model_version"] = version
    report["load_s"] = round(time.perf_counter() - t0, 2)

    datas = [p.read_bytes() for p in imgs]
    for i in range(args.warmup):
        eng.analyze(datas[i % len(datas)], filename=imgs[i % len(imgs)].name)

    walls, stages, finer, last_ok = [], {}, [], None
    for i in range(args.n):
        j = i % len(datas)
        t = time.perf_counter()
        res = eng.analyze(datas[j], filename=imgs[j].name)
        walls.append((time.perf_counter() - t) * 1000)
        for k, v in res.get("timings_ms", {}).items():
            stages.setdefault(k, []).append(v)
        if res["status"] == "ok":
            last_ok = res
            t = time.perf_counter()
            eng.finer_png(res["_state"], 0.5)
            finer.append((time.perf_counter() - t) * 1000)
    report["analyze_ms"] = stats_ms(walls)
    report["stage_ms"] = {k: stats_ms(v) for k, v in stages.items()}
    if finer:
        report["finer_rerender_ms"] = stats_ms(finer)

    if args.deep and eng.has_deep_explain:
        t = time.perf_counter()               # first call loads the FP32 weights: report it separately
        eng.deep_explain(datas[0], filename=imgs[0].name)
        report["deep_explain_cold_ms"] = round((time.perf_counter() - t) * 1000, 1)
        ds = []
        for i in range(args.deep):
            t = time.perf_counter()
            eng.deep_explain(datas[i % len(datas)], filename=imgs[i % len(imgs)].name)
            ds.append((time.perf_counter() - t) * 1000)
        report["deep_explain_ms"] = stats_ms(ds)
    else:
        report["deep_explain_ms"] = "skipped (no FP32 weights in bundle or PyTorch not installed)" if args.deep else "skipped"

    if args.llm and last_ok:
        last_ok = {k: v for k, v in last_ok.items() if k not in ("_state", "overlays")}
        report["llm"] = llm_bench(last_ok, args.llm_url, args.llm)

    report.update(peak_rss_mb=peak_rss_mb(), temp_end_c=soc_temp_c(), throttled_end=throttled(),
                  targets={"analyze_lt_ms": 400, "deep_explain_lt_ms": 6000})
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
