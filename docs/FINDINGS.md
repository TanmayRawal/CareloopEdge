# NeuroEdge — running findings log

Numbers come from the generated reports under `C:\Users\Tanmay\neuroedge_data` (manifests/*_audit.md,
runs/<run>/{metrics.json, calibration_report.md, export/quant_report.json, xai/xai_report.md}).
Model: ResNet-50, output stride 16, GAP→Linear head plus an FPN segmentation head (baseline v0, seed 0).

## 1. Data leakage in public brain-tumor MRI benchmarks
* **BRISC 2025, official split:** 566 of 1,000 test images are copies of training images, 462 of them left-right **mirrored**.
  742 of 1,000 test images come from a patient (true figshare ID) who also appears in train. Only 159 test images are clean.
* **Nickparvar v2:** 4,775 of 7,200 images are byte-identical to BRISC images.
* **figshare (Cheng 2015):** 2,873 of 3,064 images are near-duplicates of BRISC images.
* **BDNeuro-MRI (2026, described as hospital data):** 3,768 of 5,941 images are copies or mirrors of BRISC images.
* **Method:** flip-aware NCC on 64×64 thumbnails. The copy threshold of 0.98 sits in an empty gap of the distribution.
  The same-patient threshold of 0.94 was calibrated on figshare's true patient IDs (≤1 % false links). Perceptual hashes
  cannot separate copies from distinct brain slices.
* **Limitation:** image similarity recovers only ~24 % of same-patient slice pairs at 1 % false links. True patient IDs
  (carried over from figshare) cover the tumor images that came from figshare.

## 2. Class/sequence confound ("no tumor" shortcut)
* BRISC's no-tumor class (and BDNeuro's) mixes CT, FLAIR, T2 and non-contrast T1. All tumor classes are T1 post-contrast.
* Kaggle multimodal CT: tumor images carry burned-in scanner text. 512×512 images are 89 % healthy and PNG files 78 % healthy,
  so that CT model was dropped for v1.

## 3. Classification (leakage-free grouped split, seed 0)
| Set | n | Accuracy | Macro-F1 | AUROC |
|---|---|---|---|---|
| Test (grouped) | 799 | 0.949 | 0.942 | 0.994 |
| External BDNeuro (deduplicated) | 1,674 | 0.864 | 0.843 | 0.968 |
Pituitary is weakest externally: sensitivity 0.69.
Note: these are FP32 PyTorch numbers. The deployed INT8 v0 scores 0.729 macro-F1 on BDNeuro (see §10).

## 4. Deployment (INT8, ONNX Runtime QDQ; GAP and classifier kept FP32)
* 96.6 → 24.7 MB. Test macro-F1 0.939 → 0.935; McNemar p = 0.38.
* PC CPU: 28 → 11.6 ms (4 threads). Prediction agreement with FP32 is 95.7 %.
* CAM drift vs FP32: Pearson 0.93, SSIM 0.76, top-20 % IoU 0.78.
* The exact-CAM identity holds for the deployed INT8 model: decomposition error ≈ 3e-6.
* Calibration, conformal and OOD statistics must be fitted on the deployed graph. Fitting them on PyTorch outputs made
  the Mahalanobis detector flag 100 % of in-distribution images on the INT8 model; refitting on INT8 brought this to 7.4 %.

## 5. Explanations (120 correctly classified test tumors, expert masks)
* Exact Grad-CAM: 15 % of positive evidence lies inside the tumor; pointing game 0.53; ROAD 0.29.
* LayerCAM gives the best pointing game (0.76). Finer-CAM gives the best ROAD (0.29).
* ShapleyCAM with the ReST target equals Grad-CAM here. For a GAP→Linear head the ReST gradient is (2W_c − Σ p_k W_k)/hw,
  which is ≈ W_c/hw when the model is confident.
* Pituitary CAMs highlight the surrounding anatomy, not the tumor: a context shortcut. The segmentation head localizes
  the tumor itself (84 % energy inside the mask).
* Sanity check: randomizing the classifier or layer4 destroys the maps (Spearman ≈ 0). With almost all layers randomized,
  correlation returns to 0.51, so part of the spatial structure follows input intensity.
* App methods: exact Grad-CAM (default), Finer-CAM (contrastive), LayerCAM (detail view).

### 5b. Resolution ceiling and explanation-guided training (negative result)
* Test tumors are small: median 1.5 % of the image (IQR 0.9–2.7 %). Even an oracle CAM (the tumor mask itself,
  area-downsampled to the CAM grid and bilinearly upsampled) puts at most 0.55 of its energy inside the mask at 14×14
  (0.44 for small tumors) and 0.74 at 28×28. Raw energy-in-mask therefore understates CAM quality: the baseline's 0.150
  is ≈27 % of the 14×14 ceiling. LayerCAM's better pointing score partly reflects its finer layer3 grid.
* Energy-in-mask guidance loss (Rao et al. 2023), weight 1.0: the training guidance loss stayed at ≈0.85–0.94 for
  35 epochs. Test accuracy 0.937 and macro-F1 0.930, vs 0.949 and 0.942 for the baseline. Not adopted.
* Diagnosis: the guidance gradient is not small (backbone grad-norm 0.24 vs 0.44 for segmentation and 0.003 for CE).
  Optimised alone, the loss falls from 0.85 to 0.21 in 60 steps, but batch accuracy collapses to 6 %. With an exact-CAM
  GAP→Linear head the logit *is* the spatial mean of the CAM, so concentrating evidence in ~1.5 % of the image conflicts
  with classification. Classification wins in joint training.
* Future work: a segmentation-weighted pooling head, logit = Σ a_ij S_c(ij) with a from the seg map. This keeps the
  exact decomposition and makes the class decision lesion-focused by construction.

## 6. Uncertainty
* Temperature (INT8) T = 1.575; test ECE 0.049 → 0.030.
* Class-conditional conformal at α = 0.05 under-covers on test (0.87; glioma 0.59–0.73). Validation served both model
  selection and calibration, and the slices are clustered. Fix: a separate grouped calibration split in the next data version.
* Engine review triggers are calibrated on validation data, not guessed. CAM-in-lesion review threshold = 5th percentile
  of correctly classified validation tumors (0.0086). The review rate fell from 72 % to 17 %; OOD and colour images are
  rejected without a diagnosis.

## 7. Offline assistant (Qwen3-1.7B Q4_K_M via llama.cpp; grounding evaluation)
Script: `training/eval_assistant.py`. Inputs are real engine outputs on grouped-test scans. Every transcript was
read by hand: the deterministic verifier is necessary but not sufficient.
* **v1 (facts given as JSON):** the 1.7B model mostly echoed the JSON back, which passed a numbers-only check.
  After switching to prose output, ~19 % of verifier-passed answers still had a substantive error (13/68):
  misreading the 95 % conformal set ("uncertain" for a one-class set), "evidence concentrated in the lesion" at
  5–19 %, invented reasons, and "no tumor" stated for a *rejected* image.
* **Design changes:**
  * Facts are given as plain sentences with their meaning stated deterministically.
  * Questions whose answers the facts fully determine get templated answers, labelled in the UI; the LLM is
    never called. These cover the prediction set, "why X not Y", review status, treatment/outcome (refusal)
    and rejected images.
  * Verifier checks added: set and evidence contradictions, claims about image content, "95 %" outside its
    method context, "100 % confidence".
  * A "set confirms" → "set indicates" normalisation.
  * Regeneration is a fresh request at T = 0.7. A multi-turn correction made the model repeat the same draft.
* **Protocol:** rules were tuned on seed-0/1/2 samples; final numbers are on fresh cases (seed 3, 20 scans × 4
  LLM interactions; narrative, knowledge question, unknown-number bait, certainty bait).
  * Verifier: 93.8 % pass first try, 100 % after one regeneration, 0 % template fallback. Median 4.8 s,
    p95 9.2 s on the PC CPU.
  * Manual audit of those 80 passed answers: 3 clear errors (3.8 %) and 4 minor ones.
    * Clear errors: "does not rule out glioma" for a one-class set; "these features support the
      classification"; location "not available".
    * Minor: "100 % confidence"; sella location inferred rather than measured; one invented feature.
  * Verifier rules added afterwards catch all 3 clear errors and the "100 %" overclaims post hoc (6/80
    rejected, all true positives). This is post hoc, so the next fresh evaluation should confirm it.
* **Conclusion:** a 1.7B local model is reliable for short narratives and general background. It is not
  reliable for statistical interpretation, so that is deterministic by design. Human sign-off stays mandatory.

## 8. Registry deployment test (Docker Compose: FastAPI + Caddy TLS, Docker Desktop on Windows)
`server/e2e_check.py` runs against the live stack: **16/16 checks pass**, including after a forced rebuild and
recreation of the API container.
* **Passing checks:**
  * verified TLS, with clients lacking the CA refused;
  * signed and encrypted publish; immutable versions; wrong admin token refused;
  * one-time enrollment, with token reuse refused;
  * pull with trial decryption and signature check; decrypted files byte-identical to the publisher's;
    only ciphertext on the device;
  * 401 without or with forged credentials; 403 after device revocation;
  * one flipped ciphertext bit detected; another device's key cannot decrypt;
  * full audit trail.
* **Production path:** a device enrolled over TLS served the web app from its in-RAM, verified bundle
  (`edge.serve --model mri`) and returned the correct prediction.
* **Bugs found only by the live deployment** (the in-process tests passed):
  1. A NaN metric (`external_figshare.macro_auroc`: AUROC undefined on that subset) made `/latest` return
     500, because Python wrote non-standard JSON. Fix: the publisher maps non-finite values to null and sends
     with `allow_nan=False`; the server rejects such manifests with 422. Admin model-version revocation was
     added (`POST /v1/admin/models/{name}/{version}/revoke`) to withdraw the bad version.
  2. **Lost committed writes.** A host-side (Windows) read of the bind-mounted live SQLite database
     checkpointed and reset its WAL: WAL locking and shared memory do not cross the Docker Desktop VM
     boundary. A device enrollment written afterwards was silently discarded on the next container
     recreation. Fix: all registry state lives in a Docker named volume, and admin commands run inside the
     container.
* **Data at rest:** the stored blob is ciphertext (8.000 bits/byte entropy, no tar or file-name markers).

## 9. Raspberry Pi 5 deployment (8 GB, Debian 13, no Active Cooler, phone charger)
The Pi was installed with `edge/install_pi.sh`, enrolled with the registry over verified TLS, and runs the
signed, encrypted bundle decrypted in RAM. Benchmark: `edge/bench.py`, 40 held-out test scans, 4 threads.

| Measure | Pi 5 | PC (i7) | Plan target |
|---|---|---|---|
| Full analysis, median / p95 | **178 / 190 ms** | 97 ms | < 400 ms |
| INT8 model (classifier + segmentation) | 47 ms | 12 ms | — |
| Exact CAM + Finer-CAM + conformal + OOD (postprocess) | 50 ms | 27 ms | — |
| Overlay PNG rendering | 68 ms | 48 ms | — |
| Finer-CAM re-render (γ slider) | 28 ms | 20 ms | — |
| Deep Explain (FP32 LayerCAM), warm / cold | **0.78 s** / 6.5 s | 0.18 / 3.6 s | < 6 s |
| LLM prompt / generation | 45–54 / **8.2–8.9 tok/s** | 200 / 24 tok/s | 8–15 tok/s (lit.) |
| LLM narrative (≈560-token prompt, ≈120 tokens out) | ≈ 25 s | 5.8 s | — |
| Peak RSS (engine + Deep Explain) | 916 MB | 989 MB | — |

* **Thermals:** 51.8 → 83.7 °C. `get_throttled = 0xe0000`: the soft temperature limit was reached, with
  throttling and ARM-clock capping. Under-voltage did not occur (5.10 V), so the phone charger was adequate
  for this load. LLM speed fell 8.9 → 8.2 tok/s across runs as the SoC heated. **Fit the Active Cooler**
  before production or before further benchmarks; the numbers above are therefore slightly pessimistic.
* **Parity vs PC** (same INT8 bundle, 40 scans): predictions agree 40/40. Max |Δp| 0.016 (mean 0.0007);
  exact-CAM Pearson ≥ 0.998. Preprocessed inputs differ by ≤ 1.9e-5 (OpenCV 5.0 / NumPy 2.5 vs 4.13 / 2.2),
  so the probability differences come from the INT8 kernels (ARM vs x86, onnxruntime 1.30 vs 1.23), not
  from preprocessing. The verified Pi versions are locked in `edge/requirements-pi.lock`.
* **Deployment bugs found on real hardware:**
  * TLS by IP address sends no SNI, so Caddy needs `default_sni`.
  * The shared-library llama.cpp build depended on its build directory, and the unpinned web-UI download ran
    at configure time. Now a static build with the UI off.
  * The `--version`-based rebuild check never matched for shallow clones. Now a stamp file.
  * gunicorn 26's control socket hit `ProtectSystem=strict`. Now `--no-control-socket`.
  * This OS applies SD-card `network-config` only on first boot. Wi-Fi was added through a one-time
    cloud-init `runcmd`.

## 10. v1: lesion-attention pooling (exact Grad-CAM, lesion-focused by construction)
Implements the §5b future-work head. Logit z_c = b_c + Σ_ij a_ij · W_c·A_ij with a_ij = (p_ij + ε) / Σ(p + ε), where p is
the **detached** segmentation probability average-pooled to the CAM grid. The exact map S_c(ij) = a_ij · W_c·A_ij still
sums to z_c − b_c (HiResCAM form; INT8 decomposition error ~1e-7). With no lesion (p ≈ 0) the weights are uniform,
i.e. plain GAP. Other changes: split v2 with a dedicated grouped **cal** split (train 3766 / val 765 / cal 670 /
test 799 = the v0 test), focal-Dice segmentation loss with deep supervision, Grad-CAM + lesion outline only in the app.

**Ablation** (validation only, 1 seed; `RUNS/ablation_v1/summary.md`): output stride {16, 8} × ε {0.005, 0.02, 0.05}.
Pre-registered rule: val macro-F1 within 0.01 of the best, then maximum val energy-in-tumour. Chosen: **output
stride 8 (28×28 grid), ε = 0.005** (val F1 0.950, energy 0.762 = 97 % of the 28×28 oracle ceiling).

**Acceptance on test** (INT8 deployed graphs, calibration fitted on cal; `training/report_v1.py`). Deployed seed = best
validation macro-F1 (seed 0), never chosen on test.

| Metric | v0 (INT8) | v1 seed 0 (deployed) | v1 mean ± sd, 3 seeds | Gate |
|---|---|---|---|---|
| Test macro-F1 [95 % CI] | 0.935 [0.917, 0.951] | **0.944** [0.926, 0.959] | 0.938 ± 0.007 | pass |
| External BDNeuro macro-F1 | 0.729 | **0.836** | 0.827 ± 0.012 | pass |
| Grad-CAM energy in tumour | 0.150 | **0.728** (98 % of ceiling 0.746) | 0.713 ± 0.018 | pass |
| Pointing game | 0.525 | **0.992** | 0.981 ± 0.019 | pass |
| ROAD (combined) | 0.288 | **0.324** | 0.328 ± 0.005 | pass |
| Conformal coverage, α = 0.05 | 0.867 | **0.911** | 0.917 ± 0.006 | **fail (target 0.93)** |

* Paired test comparison, v1 seed 0 vs v0 on the identical 799 images: ΔF1 = +0.009 [−0.005, +0.025]; McNemar
  21 vs 13 discordant, p = 0.23. Non-inferior; not shown to be better.
* Per-class Grad-CAM energy in tumour: glioma 0.635, meningioma 0.844, **pituitary 0.704** (v0: anatomy, not tumour).
* **Correction to §3:** the 0.843 external macro-F1 was the FP32 PyTorch model. The deployed v0 INT8 graph scores
  **0.729** on BDNeuro (accuracy 0.783), an 11-point quantisation loss on shifted data that the in-distribution INT8
  check (§4) did not reveal. v1 loses little: FP32 0.842 → INT8 0.836 (seed 0). External INT8 evaluation is now part
  of acceptance.
* **Coverage** improved (0.867 → 0.911) but misses the pre-registered 0.93; per class glioma 0.828, meningioma 0.927,
  no_tumor 0.988, pituitary 0.891. Coverage is 0.960 on cal itself and 0.933 on val, so the BRISC test images are not
  exchangeable with the cal images (test comes from the dataset's own test release). The threshold was not re-tuned
  after seeing test; v1 is deployed with this documented limitation. Fix (next data version): calibrate per source /
  release (Mondrian) or include test-release-like scans in cal.
* **Sanity check** (cascading randomization): randomizing the classifier gives Spearman −0.04 (destroyed), but SSIM
  stays 0.81 (v0: 0.39). The spatial envelope of the map comes partly from the segmentation-derived attention, so the
  map is "lesion-weighted classifier evidence", not classifier evidence alone. The rank order inside it still depends
  on the classifier. With all layers randomized, correlation returns to 0.57 (v0 0.51): input-intensity structure.
* **Pi 5** (seed 0 INT8, 60 scans, 4 threads): full analysis median 238 ms / p95 250 ms (v0: 178 ms; target < 400 ms);
  model 120 ms; peak RSS 261 MB; 43 → 63 °C, no throttling.
