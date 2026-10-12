"""v1 acceptance report (plan A6): the 3 v1 seeds vs the deployed v0 model, on the deployed INT8 graphs.

  python -m training.report_v1 [--v1 RUNS/mri_grouped_v1_os8_eps0.005] [--v0 RUNS/mri_grouped/seed0]

Reads, per run, export/calib_model_int8/{pred_*.npz, calibration.json} (written by `calibrate --onnx`) and
xai/xai_report.json (written by `xai_eval --split test`). Test = the v0 test split (same 799 images), touched once.
The deployed seed is the one with the best VALIDATION macro-F1 (metrics.json), never chosen on test.
Writes  <v1>/acceptance.json  and  <v1>/acceptance.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.data.paths import DATA_ROOT
from neuroedge.metrics import bootstrap_ci, mcnemar_exact

GRADCAM_KEYS = ("Grad-CAM (exact, HiResCAM form)", "CAM (exact) / Grad-CAM")


def macro_f1(y: np.ndarray, p: np.ndarray, k: int) -> float:
    f = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f))


def load_preds(run: Path, split: str):
    z = np.load(run / "export" / "calib_model_int8" / f"pred_{split}.npz", allow_pickle=True)
    return z["labels"].astype(int), z["logits"].argmax(1), np.asarray(z["ids"]).astype(str)


def summarise(run: Path) -> dict:
    cal = json.loads((run / "export" / "calib_model_int8" / "calibration.json").read_text())
    k = len(cal["class_names"])
    out = {"run": str(run)}
    for split in ("test", "external_bdneuro"):
        y, p, _ = load_preds(run, split)
        f1, lo, hi = bootstrap_ci(lambda i: macro_f1(y[i], p[i], k), len(y))
        out[split] = {"n": int(len(y)), "macro_f1": f1, "f1_lo": lo, "f1_hi": hi, "acc": float(np.mean(y == p))}
    t = cal["splits"]["test"]
    out["test"]["coverage"] = t["coverage"]["marginal_coverage"]
    out["test"]["per_class_coverage"] = {cal["class_names"][int(c)]: v
                                         for c, v in t["coverage"]["per_class_coverage"].items()}
    out["test"]["ece_after"] = t["ece_after"]
    xai = json.loads((run / "xai" / "xai_report.json").read_text())
    assert xai["split"] == "test", f"{run}: XAI report is for split {xai['split']}"
    key = next(k_ for k_ in GRADCAM_KEYS if k_ in xai["methods"])
    m = xai["methods"][key]
    out["xai"] = {name: m[name] for name in ("energy_pointing", "pointing_game", "road_combined") if name in m}
    for extra in ("energy_per_class", "gradcam_fraction_of_ceiling", "oracle_energy"):
        if extra in xai:
            out["xai"][extra] = xai[extra]
    out["sanity"] = xai.get("sanity_cascading_randomization")
    metrics = run / "metrics.json"
    if metrics.exists():
        out["val_macro_f1"] = json.loads(metrics.read_text())["val"]["macro_f1"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", type=Path, default=DATA_ROOT / "runs" / "mri_grouped_v1_os8_eps0.005")
    ap.add_argument("--v0", type=Path, default=DATA_ROOT / "runs" / "mri_grouped" / "seed0")
    a = ap.parse_args()

    v0 = summarise(a.v0)
    seeds = {d.name: summarise(d) for d in sorted(a.v1.glob("seed*"))}
    best = max(seeds, key=lambda s: seeds[s]["val_macro_f1"])

    def ms(get):
        v = np.array([get(s) for s in seeds.values()], float)
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "values": v.tolist()}

    agg = {
        "test_macro_f1": ms(lambda s: s["test"]["macro_f1"]),
        "external_bdneuro_macro_f1": ms(lambda s: s["external_bdneuro"]["macro_f1"]),
        "test_coverage": ms(lambda s: s["test"]["coverage"]),
        "energy_in_tumour": ms(lambda s: s["xai"]["energy_pointing"]["mean"]),
        "pointing_game": ms(lambda s: s["xai"]["pointing_game"]["mean"]),
        "road": ms(lambda s: s["xai"]["road_combined"]["mean"]),
    }
    # paired comparison, deployed v1 seed vs v0, on the identical test images (matched by id)
    y0, p0, i0 = load_preds(a.v0, "test")
    y1, p1, i1 = load_preds(a.v1 / best, "test")
    pos = {k: n for n, k in enumerate(i1)}
    idx = np.array([pos[k] for k in i0])
    assert np.array_equal(y0, y1[idx]), "test labels differ between v0 and v1"
    p1 = p1[idx]
    k = 4
    d, dlo, dhi = bootstrap_ci(lambda i: macro_f1(y0[i], p1[i], k) - macro_f1(y0[i], p0[i], k), len(y0))
    paired = {"delta_macro_f1": d, "lo": dlo, "hi": dhi, "mcnemar": mcnemar_exact(p1 == y0, p0 == y0)}

    B = seeds[best]
    gates = {
        "test_macro_f1 >= v0 - 0.01 (CI non-inferiority: delta CI lower bound > -0.01)": dlo > -0.01,
        "external_bdneuro >= v0 - 0.01": B["external_bdneuro"]["macro_f1"] >= v0["external_bdneuro"]["macro_f1"] - 0.01,
        "energy_in_tumour >= 0.30": B["xai"]["energy_pointing"]["mean"] >= 0.30,
        "pointing_game >= 0.70": B["xai"]["pointing_game"]["mean"] >= 0.70,
        "ROAD >= v0": B["xai"]["road_combined"]["mean"] >= v0["xai"]["road_combined"]["mean"],
        "test coverage >= 0.93": B["test"]["coverage"] >= 0.93,
    }
    report = {"v0": v0, "seeds": seeds, "deployed_seed": best, "aggregate": agg, "paired_vs_v0": paired,
              "gates": gates, "accepted": all(gates.values())}
    (a.v1 / "acceptance.json").write_text(json.dumps(report, indent=2, default=float))

    f = lambda x: f"{x:.3f}"  # noqa: E731
    L = [f"# v1 acceptance (test split, INT8 deployed graphs)\n",
         f"Deployed seed (best validation macro-F1): **{best}**\n",
         "| metric | v0 | v1 deployed | v1 mean ± sd (3 seeds) | gate |", "|---|---|---|---|---|"]
    rows = [("Test macro-F1", v0["test"]["macro_f1"], B["test"]["macro_f1"], agg["test_macro_f1"], list(gates.values())[0]),
            ("External BDNeuro macro-F1", v0["external_bdneuro"]["macro_f1"], B["external_bdneuro"]["macro_f1"],
             agg["external_bdneuro_macro_f1"], list(gates.values())[1]),
            ("Grad-CAM energy in tumour", v0["xai"]["energy_pointing"]["mean"], B["xai"]["energy_pointing"]["mean"],
             agg["energy_in_tumour"], list(gates.values())[2]),
            ("Pointing game", v0["xai"]["pointing_game"]["mean"], B["xai"]["pointing_game"]["mean"],
             agg["pointing_game"], list(gates.values())[3]),
            ("ROAD", v0["xai"]["road_combined"]["mean"], B["xai"]["road_combined"]["mean"], agg["road"],
             list(gates.values())[4]),
            ("Conformal coverage (test)", v0["test"]["coverage"], B["test"]["coverage"], agg["test_coverage"],
             list(gates.values())[5])]
    for name, a0, a1, m, ok in rows:
        L.append(f"| {name} | {f(a0)} | {f(a1)} | {f(m['mean'])} ± {f(m['sd'])} | {'PASS' if ok else 'FAIL'} |")
    L += ["", f"Test macro-F1 95% CI: v0 [{f(v0['test']['f1_lo'])}, {f(v0['test']['f1_hi'])}], "
              f"v1 [{f(B['test']['f1_lo'])}, {f(B['test']['f1_hi'])}]",
          f"Paired ΔF1 (v1 − v0) = {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]; McNemar {paired['mcnemar']}",
          f"Per-class test coverage (v1): " + ", ".join(f"{c} {v:.3f}" for c, v in B["test"]["per_class_coverage"].items()),
          "", f"**Accepted: {report['accepted']}**"]
    (a.v1 / "acceptance.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
