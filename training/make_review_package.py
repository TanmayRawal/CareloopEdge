"""Build a self-contained review zip for an external reviewer (code + trained model + sample scans + run scripts).

  python -m training.make_review_package [--out DIR]

Contents (secrets, datasets, patient data and the 1.1 GB LLM are never included):
  code/           `git archive HEAD` (exactly what is committed; ignored files and secrets cannot leak in)
  model/          the INT8 v1 bundle the portal serves (mri/0.2.0)
  sample_scans/   held-out BRISC 2025 test slices (CC BY 4.0, attribution file included)
  README.md, run_demo.bat / run_demo.sh, requirements-demo.txt, SHA256SUMS.txt
The demo portal's database, encryption key and test accounts are created on the reviewer's machine at first start.
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
BUNDLE = DATA_ROOT / "bundles" / "mri_v1_0.2.0_int8"
MODEL_DIR = "mri_v1"
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
argon2-cffi
"""

RUN_BAT = r"""@echo off
rem NeuroEdge review demo (Windows). Double-click, or run from a terminal.
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
echo.
echo NeuroEdge is starting: open http://localhost:8080 (Ctrl+C here to stop)
echo Test logins (patient, doctor, admin) are in portal_data\DEV_ACCOUNTS.txt
start "" http://localhost:8080
cd code
python -m edge.serve --bundle-dir ..\model\mri_v1 --portal-data ..\portal_data --init-dev --port 8080
goto :eof
:err
echo.
echo Setup failed. Make sure 64-bit Python 3.10-3.13 is installed (python.org) and try again.
pause
"""

RUN_SH = """#!/usr/bin/env bash
# NeuroEdge review demo (macOS / Linux):  bash run_demo.sh
set -e
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
[ -d .venv ] || { echo "Creating a private Python environment in .venv ..."; "$PY" -m venv .venv; }
. .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
echo "Installing packages (first run only, ~150 MB) ..."
python -m pip install -r requirements-demo.txt
echo; echo "NeuroEdge is starting: open http://localhost:8080 (Ctrl+C to stop)"
echo "Test logins (patient, doctor, admin) are in portal_data/DEV_ACCOUNTS.txt"
( sleep 3; (command -v open >/dev/null && open http://localhost:8080) || (command -v xdg-open >/dev/null && xdg-open http://localhost:8080) ) >/dev/null 2>&1 &
cd code
exec python -m edge.serve --bundle-dir ../model/mri_v1 --portal-data ../portal_data --init-dev --port 8080
"""

README = """# NeuroEdge: review package (v2 portal, model mri/0.2.0)

Secure, explainable brain-tumor MRI review that runs offline on a Raspberry Pi 5. This package runs the same
clinic portal and trained model **on your laptop** so you can review the workflow, code, results and documentation.

> Research prototype on public data; not a medical device.

## 1. Run it (about 5 minutes, internet needed only the first time)
Requirements: 64-bit **Python 3.10–3.13** (python.org) and about 1 GB of free disk space.

| System | Command |
|---|---|
| Windows | double-click **`run_demo.bat`** |
| macOS / Linux | `bash run_demo.sh` in a terminal opened in this folder |

The script:
1. creates a private environment in `.venv` (nothing is installed system-wide);
2. installs the packages;
3. on first start, creates a local demo database with **test accounts**, listed in `portal_data/DEV_ACCOUNTS.txt`
   (patient, doctor, admin);
4. opens **http://localhost:8080**.

Stop the app with Ctrl+C. On the Pi the same portal is served over HTTPS; on the laptop it is plain localhost.

**Try the workflow**
1. Sign in as the **patient**, upload a scan from `sample_scans/` (the file names start with the true class), and
   sign out. The patient sees only "waiting for your doctor", never AI output.
2. Sign in as the **doctor**. The case appears on the prioritised worklist. Open it and look at:
   * the calibrated probabilities and the **95% prediction set**;
   * **Grad-CAM** (exact for this architecture, lesion-focused) and the **lesion outline**;
   * the review flags and the assistant.
3. Pick the final diagnosis and **sign the report** (it asks for your password again).
4. Sign in as the patient again: the doctor's signed report is now visible.
5. Book a consultation as the patient (from the signed report), then open **Schedule** as the doctor to mark
   attendance. The demo radiologist works Mon–Sat 09:00–13:00 and 14:00–17:00 (change under **Availability**).
6. The **admin** account has users, department settings, the e-mail outbox and the audit log. In this demo
   every e-mail (confirmations, reminders) is saved as an `.eml` file in `portal_data/mail/`; open one with any
   mail program.

**Optional: AI assistant narratives.** These need a local LLM server; without it the assistant uses deterministic
template answers. To enable it:
1. Download a llama.cpp release for your OS from https://github.com/ggml-org/llama.cpp/releases.
2. Download the model from https://huggingface.co/unsloth/Qwen3-1.7B-GGUF (file `Qwen3-1.7B-Q4_K_M.gguf`,
   SHA-256 `b139949c…1897`).
3. Start `llama-server -m Qwen3-1.7B-Q4_K_M.gguf -c 4096 --port 8081` before the demo.

## 2. What to review
| Where | What |
|---|---|
| `code/README.md` | Architecture overview and results |
| `code/docs/MODEL_CARD.md` | Data, metrics, explanations, intended use, limitations |
| `code/docs/FINDINGS.md` | All experiments and measurements, including negative results and bugs found (§10 = v1) |
| `code/docs/PI_DEPLOY.md` | Secure server and Raspberry Pi deployment |
| `code/neuroedge/models.py`, `code/neuroedge/cam.py` | Lesion-attention pooling head and exact Grad-CAM maths |
| `code/neuroedge/uncertainty.py` | Temperature scaling, conformal prediction, Mahalanobis OOD |
| `code/training/data/`, `code/training/ablation_v1.py`, `code/training/report_v1.py` | Leakage audit and split, ablation, acceptance report |
| `code/edge/portal/` | Clinic portal: roles, encrypted storage, signed reports, audit |
| `code/neuroedge/security.py`, `code/server/` | Signing, encryption, registry |
| `code/tests/` | 108 tests. Running them needs the training dependencies (`code/requirements-train.txt`, PyTorch). |

## 3. Key results (deployed INT8 model, held-out leakage-free test set)
| | v0 | v1 (3-seed mean ± sd) |
|---|---|---|
| Test macro-F1 (n = 799) | 0.935 | **0.944** (0.938 ± 0.007) |
| External BDNeuro macro-F1 (deduplicated, n = 1,674) | 0.729 | **0.836** (0.827 ± 0.012) |
| Grad-CAM evidence inside the tumor | 0.15 | **0.73** |
| Pointing game | 0.53 | **0.99** |
| 95% prediction-set coverage | 0.87 | **0.91** (target 0.93 not met) |

* Raspberry Pi 5: 238 ms per analysis; 40/40 predictions identical to the PC.
* The official BRISC test split leaks: 566 of 1,000 test images are copies or mirrors of training images. All
  numbers above use a re-split by patient.
* Known limitations are in the model card: the sequence shortcut in the "no tumor" class, coverage below the
  target, and single 2D slices.

## 4. Contents and integrity
* `code/` is the committed source (git {commit}).
* `model/` is the trained v1 bundle.
* `sample_scans/` holds 40 held-out test slices.
* `SHA256SUMS.txt` lists a checksum for every file.
* Datasets, private keys, patient data and the LLM weights are not included.

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
    args = ap.parse_args()

    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip():
        print("WARNING: uncommitted changes are NOT in the package (it uses the last commit).")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    date = time.strftime("%Y-%m-%d")
    name = "NeuroEdge_review"                              # short top folder (Windows path limit)
    zip_name = f"NeuroEdge_review_v2_{date}_{commit}"
    stage = args.out / name
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "code").mkdir(parents=True)

    tar = stage / "code.tar"
    subprocess.run(["git", "archive", "--format=tar", "-o", str(tar), "HEAD"], cwd=ROOT, check=True)
    shutil.unpack_archive(tar, stage / "code")
    tar.unlink()
    # Windows' 260-character path limit: the original notebook's long name broke "Extract All" in testing
    for nb in (stage / "code" / "reference").glob("*.ipynb"):
        nb.rename(nb.with_name("original_notebook.ipynb"))

    mdir = stage / "model" / MODEL_DIR
    mdir.mkdir(parents=True)
    for f in ("bundle.json", "model.onnx", "head.npz", "calibration.json", "ood.npz"):
        shutil.copy2(BUNDLE / f, mdir / f)
    shutil.copytree(SAMPLES, stage / "sample_scans")
    # write_bytes, not write_text: on Windows write_text emits CRLF, which breaks SHA256SUMS.txt (sha256sum -c)
    lf = lambda path, text: path.write_bytes(text.encode("utf-8"))  # noqa: E731
    lf(stage / "sample_scans" / "ATTRIBUTION.txt",
       "Held-out test slices from BRISC 2025 (arXiv:2506.14318), CC BY 4.0.\n"
       "File names start with the ground-truth class.\n")

    lf(stage / "requirements-demo.txt", REQ_DEMO)
    (stage / "run_demo.bat").write_bytes(RUN_BAT.replace("\n", "\r\n").encode())
    lf(stage / "run_demo.sh", RUN_SH)
    lf(stage / "README.md", README.replace("{commit}", commit).replace("{date}", date))

    files = sorted(p for p in stage.rglob("*") if p.is_file())
    lf(stage / "SHA256SUMS.txt", "".join(f"{sha256(p)}  {p.relative_to(stage).as_posix()}\n" for p in files))

    zpath = args.out / f"{zip_name}.zip"
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
