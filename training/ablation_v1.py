"""v1 ablation: output stride {16, 8} x attention eps {0.005, 0.02, 0.05}, seed 0, lesion-attention head.

  python -m training.ablation_v1                 # resumable: finished steps are skipped

For every variant: train -> export -> INT8 -> XAI evaluation on the VALIDATION split. Selection uses only
validation data (the test split is reserved for the final 3-seed model):
  1. keep variants whose val macro-F1 >= best val macro-F1 - 0.01   (classification non-inferiority)
  2. among them, highest val Grad-CAM energy-in-tumour (ties: pointing game)
Writes <RUNS>/ablation_v1/summary.{json,md}.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from training.data.paths import RUNS

CONFIG = "training/configs/mri_v1.yaml"
VARIANTS = [(f"os{os_}_eps{eps}", os_, eps) for os_ in (16, 8) for eps in (0.005, 0.02, 0.05)]
PY = sys.executable


def step(cmd: list[str], done: Path, log: Path) -> None:
    if done.exists():
        print(f"  skip (done): {done.name}", flush=True)
        return
    print("  $ " + " ".join(cmd[2:]), flush=True)
    with open(log, "a", encoding="utf-8") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    if r.returncode != 0 or not done.exists():
        raise SystemExit(f"step failed ({r.returncode}); see {log}")


def collect(name: str, run: Path) -> dict:
    m = json.loads((run / "metrics.json").read_text())
    q = json.loads((run / "export" / "quant_report.json").read_text())
    x = json.loads((run / "xai" / "xai_report.json").read_text())
    cam = x["methods"]["Grad-CAM (exact, HiResCAM form)"]
    return {"variant": name, "best_epoch": m["best_epoch"],
            "val_macro_f1": m["val"]["macro_f1"], "val_acc": m["val"]["accuracy"],
            "val_seg_dice": m["val"].get("seg_dice_mean"),
            "val_int8_macro_f1": q["val"]["int8"]["macro_f1"], "int8_cam_pearson": q["val"]["cam_drift"]["pearson"],
            "int8_decomp_err": q["val"]["int8_cam_decomposition_error"],
            "pc_int8_4thr_ms": q.get("pc_cpu_latency_ms", {}).get("int8_4thr"),
            "xai_energy": cam["energy_pointing"]["mean"], "xai_energy_ci": [cam["energy_pointing"]["lo"], cam["energy_pointing"]["hi"]],
            "xai_pointing": cam["pointing_game"]["mean"], "xai_road": cam["road_combined"]["mean"],
            "xai_energy_per_class": {k: v["mean"] for k, v in cam["energy_per_class"].items()},
            "oracle_ceiling": x["oracle_energy_ceiling"]["mean"], "fraction_of_ceiling": x["gradcam_fraction_of_ceiling"],
            "sanity_spearman_after_layer4": x["sanity_cascading_randomization"][1]["spearman"]["mean"]}


def main() -> None:
    out = RUNS / "ablation_v1"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, os_, eps in VARIANTS:
        run = RUNS / f"mri_grouped_v1_{name}" / "seed0"
        log = out / f"{name}.log"
        print(f"== {name}", flush=True)
        step([PY, "-m", "training.train", "--config", CONFIG, "--seed", "0", "--tag", f"_v1_{name}",
              "--set", f"model.output_stride={os_}", "--set", f"model.attn_eps={eps}"], run / "metrics.json", log)
        step([PY, "-m", "training.export", "--run", str(run)], run / "export" / "bundle.json", log)
        step([PY, "-m", "training.quantize", "--run", str(run)], run / "export" / "quant_report.json", log)
        step([PY, "-m", "training.xai_eval", "--run", str(run), "--split", "val", "--n-per-class", "40"],
             run / "xai" / "xai_report.json", log)
        rows.append(collect(name, run))
        (out / "summary.json").write_text(json.dumps(rows, indent=2))

    best_f1 = max(r["val_macro_f1"] for r in rows)
    eligible = [r for r in rows if r["val_macro_f1"] >= best_f1 - 0.01]
    chosen = max(eligible, key=lambda r: (r["xai_energy"], r["xai_pointing"]))
    summary = {"rule": "val macro-F1 within 0.01 of best, then max val energy-in-tumour (tie: pointing)",
               "chosen": chosen["variant"], "variants": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    L = ["# v1 ablation (validation split only, seed 0)", "",
         "| variant | val F1 | val F1 INT8 | seg Dice | energy in tumour | % of ceiling | pointing | ROAD | INT8 CAM r | INT8 ms (PC) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['variant']}{' **(chosen)**' if r is chosen else ''} | {r['val_macro_f1']:.3f} | {r['val_int8_macro_f1']:.3f} | "
                 f"{(r['val_seg_dice'] or float('nan')):.3f} | {r['xai_energy']:.3f} | {r['fraction_of_ceiling']:.0%} | "
                 f"{r['xai_pointing']:.2f} | {r['xai_road']:.3f} | {r['int8_cam_pearson']:.2f} | {r['pc_int8_4thr_ms'] or float('nan'):.1f} |")
    L += ["", f"Selection rule: {summary['rule']}. Chosen: **{chosen['variant']}**.", "",
          "Per-class energy in tumour (chosen): " + ", ".join(f"{k} {v:.3f}" for k, v in chosen["xai_energy_per_class"].items())]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
