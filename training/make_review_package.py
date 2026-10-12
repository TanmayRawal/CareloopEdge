"""Build a self-contained review zip for an external reviewer (code + trained model + sample scans + run scripts).

  python -m training.make_review_package [--no-deep-explain] [--out DIR]

Contents (secrets, datasets and the 1.1 GB LLM are never included):
  code/           `git archive HEAD` (exactly what is committed; ignored files and secrets cannot leak in)
  model/          the INT8 bundle the app serves (+ FP32 weights for Deep Explain unless --no-deep-explain)
  sample_scans/   held-out BRISC 2025 test slices (CC BY 4.0, attribution file included)
  README.md, run_demo.bat / run_demo.sh, requirements-demo.txt, requirements-deep.txt, SHA256SUMS.txt
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from training.data.paths import DATA_ROOT

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = DATA_ROOT / "bundles" / "mri_v0_int8"
SAMPLES = DATA_ROOT / "bench_samples"

REQ_DEMO = """# Laptop demo (Windows / macOS / Linux, 64-bit Python 3.10-3.13)
numpy
opencv-python-headless
onnxruntime
pydicom
pyyaml
scipy
cryptography
flask
"""
REQ_DEEP = """# Optional: enables the "Deep Explain" (FP32 LayerCAM) tab. CPU-only PyTorch is enough.
torch
timm
"""

RUN_BAT = r"""@echo off
rem NeuroEdge review demo (Windows). Double-click, or run from a terminal. Add "deep" to also install Deep Explain.
cd /d "%~dp0"
set PY=python
where py >nul 2>nul && set PY=py -3
if not exist .venv (
  echo Creating a private Python environment in .venv ...
  %PY% -m venv .venv || goto :err
)
call .venv\Scripts\activate.bat || goto :err
python -m pip install --upgrade pip >nul
echo Installing packages (first run only, ~150 MB) ...
python -m pip install -r requirements-demo.txt || goto :err
if /i "%1"=="deep" python -m pip install -r requirements-deep.txt --extra-index-url https://download.pytorch.org/whl/cpu || goto :err
echo.
echo NeuroEdge is starting: open http://localhost:8080 (Ctrl+C here to stop)
start "" http://localhost:8080
cd code
python -m edge.serve --bundle-dir ..\model\mri_v0_int8 --port 8080
goto :eof
:err
echo.
echo Setup failed. Make sure 64-bit Python 3.10-3.13 is installed (python.org) and try again.
pause
"""

RUN_SH = """#!/usr/bin/env bash
# NeuroEdge review demo (macOS / Linux):  bash run_demo.sh      (add "deep" to also install Deep Explain)
set -e
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
[ -d .venv ] || { echo "Creating a private Python environment in .venv ..."; "$PY" -m venv .venv; }
. .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
echo "Installing packages (first run only, ~150 MB) ..."
python -m pip install -r requirements-demo.txt
if [ "$1" = "deep" ]; then python -m pip install -r requirements-deep.txt --extra-index-url https://download.pytorch.org/whl/cpu; fi
echo; echo "NeuroEdge is starting: open http://localhost:8080 (Ctrl+C to stop)"
( sleep 3; (command -v open >/dev/null && open http://localhost:8080) || (command -v xdg-open >/dev/null && xdg-open http://localhost:8080) ) >/dev/null 2>&1 &
cd code
exec python -m edge.serve --bundle-dir ../model/mri_v0_int8 --port 8080
"""

README = """# NeuroEdge: review package

Secure, explainable brain-tumor MRI classification that runs offline on a Raspberry Pi 5. This package lets you
run the same web app and trained model **on your laptop** and review the code, results and documentation.

> **Research and education only. Not a medical device.** Outputs must not be used for diagnosis or treatment.

## 1. Run it (about 5 minutes, internet needed only the first time)
Requirements: 64-bit **Python 3.10–3.13** (python.org) and about 1 GB of free disk space.

| System | Command |
|---|---|
| Windows | double-click **`run_demo.bat`** |
| macOS / Linux | `bash run_demo.sh` in a terminal opened in this folder |

The script creates a private environment in `.venv` (nothing is installed system-wide), installs the packages,
starts the app and opens **http://localhost:8080**. Upload any scan from `sample_scans/`; the file names start
with the true class. Stop the app with Ctrl+C.

What to look at on a result page:
* **Calibrated probabilities** and the **95% prediction set** (conformal). Odd inputs are rejected, not classified.
* **Grad-CAM** (exact for this architecture), **Finer-CAM** (with a slider for comparison strength γ), and the
  **lesion outline** from the separate segmentation head.
* **Draft report** (template report with a reviewer sign-off) and **Ask about this result**.

Optional extras:
* **Deep Explain** (FP32 LayerCAM): run `run_demo.bat deep` or `bash run_demo.sh deep`. This adds CPU PyTorch,
  about 300 MB.
* **AI assistant narratives**: these need a local LLM server. Without it the app uses deterministic template
  answers. To enable it:
  1. Download a llama.cpp release for your OS from https://github.com/ggml-org/llama.cpp/releases.
  2. Download the model from https://huggingface.co/unsloth/Qwen3-1.7B-GGUF (file `Qwen3-1.7B-Q4_K_M.gguf`,
     SHA-256 `b139949c…1897`).
  3. Start `llama-server -m Qwen3-1.7B-Q4_K_M.gguf -c 4096 --port 8081` before the demo.

## 2. What to review
| Where | What |
|---|---|
| `code/README.md` | Architecture overview and results |
| `code/docs/MODEL_CARD.md` | Data, metrics, explanations, intended use, limitations |
| `code/docs/FINDINGS.md` | All experiments and measurements, including negative results and bugs found |
| `code/docs/PI_DEPLOY.md` | Secure server and Raspberry Pi deployment |
| `code/neuroedge/cam.py` | Exact CAM / Finer-CAM maths |
| `code/neuroedge/uncertainty.py` | Temperature scaling, conformal prediction, Mahalanobis OOD |
| `code/training/data/` | Leakage audit and patient-grouped split |
| `code/neuroedge/security.py`, `code/server/` | Signing, encryption, registry |
| `code/tests/` | 75 tests. Running them needs the training dependencies (`code/requirements-train.txt`, PyTorch). |

## 3. Key results (held-out, leakage-free)
| Set | Accuracy | Macro-F1 | Macro-AUROC |
|---|---|---|---|
| Test (patient-grouped BRISC split, n = 799) | 0.949 | 0.942 | 0.994 |
| Test, INT8 model as deployed | 0.939 | 0.935 | 0.991 |
| External BDNeuro (deduplicated, n = 1,674) | 0.864 | 0.843 | 0.968 |

* Raspberry Pi 5: 178 ms per analysis; 40/40 predictions identical to the PC.
* The official BRISC test split leaks: 566 of 1,000 test images are copies or mirrors of training images. All
  numbers above use a re-split by patient.
* Known limitations (sequence shortcut in the "no tumor" class, pituitary context shortcut, conformal
  under-coverage, single seed) are in the model card.

## 4. Contents and integrity
`code/` is the committed source (git {commit}), `model/` the trained bundle, and `sample_scans/` 40 held-out
test slices. `SHA256SUMS.txt` lists a checksum for every file. Datasets, private keys and the LLM weights are
not included.

**Data attribution:** the sample scans are from the BRISC 2025 dataset (arXiv:2506.14318),
licensed CC BY 4.0.

Built {date}.
"""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "export")
    ap.add_argument("--no-deep-explain", action="store_true", help="omit the 97 MB FP32 weights")
    args = ap.parse_args()

    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip():
        print("WARNING: uncommitted changes are NOT in the package (it uses the last commit).")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    date = time.strftime("%Y-%m-%d")
    name = f"NeuroEdge_review_{date}_{commit}"
    stage = args.out / name
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "code").mkdir(parents=True)

    tar = stage / "code.tar"
    subprocess.run(["git", "archive", "--format=tar", "-o", str(tar), "HEAD"], cwd=ROOT, check=True)
    shutil.unpack_archive(tar, stage / "code")
    tar.unlink()

    mdir = stage / "model" / "mri_v0_int8"
    mdir.mkdir(parents=True)
    for f in ("bundle.json", "model.onnx", "head.npz", "calibration.json", "ood.npz") + \
             (() if args.no_deep_explain else ("model_fp32.pt",)):
        shutil.copy2(BUNDLE / f, mdir / f)
    shutil.copytree(SAMPLES, stage / "sample_scans")
    (stage / "sample_scans" / "ATTRIBUTION.txt").write_text(
        "Held-out test slices from BRISC 2025 (arXiv:2506.14318), CC BY 4.0.\n"
        "File names start with the ground-truth class.\n")

    (stage / "requirements-demo.txt").write_text(REQ_DEMO)
    (stage / "requirements-deep.txt").write_text(REQ_DEEP)
    (stage / "run_demo.bat").write_bytes(RUN_BAT.replace("\n", "\r\n").encode())
    (stage / "run_demo.sh").write_bytes(RUN_SH.encode())
    (stage / "README.md").write_text(README.replace("{commit}", commit).replace("{date}", date), encoding="utf-8")

    files = sorted(p for p in stage.rglob("*") if p.is_file())
    (stage / "SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.relative_to(stage).as_posix()}\n" for p in files))

    zpath = args.out / f"{name}.zip"
    zpath.unlink(missing_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                info = zipfile.ZipInfo.from_file(p, f"{name}/{p.relative_to(stage).as_posix()}")
                if p.suffix == ".sh":
                    info.external_attr = (0o755 << 16)
                with open(p, "rb") as f:
                    z.writestr(info, f.read(), zipfile.ZIP_DEFLATED)
    shutil.rmtree(stage)
    print(f"{zpath}  ({zpath.stat().st_size / 2**20:.1f} MB, {len(files) + 1} files, commit {commit})")


if __name__ == "__main__":
    main()
