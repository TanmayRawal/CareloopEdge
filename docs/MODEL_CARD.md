# Model card: NeuroEdge MRI (current: registry version `mri/0.3.0`)

> Research prototype on public data, not a medical device. In the portal, every result is reviewed and signed
> by a doctor before a patient sees anything; patients never see AI output.

## `mri/0.3.0` (current): whole-study model
* **What changed:** same architecture, preprocessing (same fingerprint) and training recipe as `mri/0.2.0` below. The
  training data adds slices from 3D patient studies (BraTS 2023 GLI and MEN, 300 each; NHNN pituitary, 60), three
  planes, with tumour-free slices of the same scans as "no tumor". Only train/val/cal patients are used; test patients
  are held out by patient.
* **Whole-study mode:** 36 slices (12 per plane) -> per-slice INT8 analysis -> mean of the 5 slices most suggestive
  of tumour (`topk`, chosen on calibration patients) -> one answer per study; Grad-CAM and the lesion outline for the 3
  key slices. Single images still work as before.
* **Held-out patients (pre-registered targets):**
  * tumour-type macro-F1: 0.98 (target ≥ 0.85; v1: 0.71);
  * slice tumour AUROC: 0.95 (target ≥ 0.90; v1: 0.72);
  * IXI healthy false-alarm rate: 5% (target ≤ 10%; v1: 23%);
  * accuracy per dataset: glioma 100% (76), meningioma 91% (68), pituitary 100% (15), healthy 95% (380).
* **2D test sets:** BRISC macro-F1 0.919 (v1: 0.944); INT8 vs FP32 within 1 point (McNemar p = 1.0).
* **Limits:**
  * one training seed;
  * the test patients come from the same datasets as the training patients; the external whole-study test
    (BraTS-Africa) is pending;
  * IXI is non-contrast T1;
  * the 95% set is calibrated on slices, not on patients;
  * BraTS data is CC BY-NC: research and non-commercial use only.

The sections below describe the shared architecture and the single-slice `mri/0.2.0` release.

## Model
| | |
|---|---|
| Task | 4-class classification of a single 2D brain MRI slice (glioma, meningioma, no tumor, pituitary tumor) plus a tumor segmentation mask |
| Architecture | ResNet-50 (timm, ImageNet init), output stride 8 (28×28 feature map), **lesion-attention pooling** → Linear head, FPN segmentation head with deep supervision |
| Lesion-attention pooling | z_c = b_c + Σ_ij a_ij · W_c·A_ij, with a_ij = (p_ij + ε) / Σ(p + ε). p is the **detached** segmentation probability pooled to the grid; ε = 0.005. With no lesion the weights are uniform (plain GAP). Classification cannot reshape the mask. |
| Training | Cross-entropy + focal BCE + Dice (segmentation terms only where a mask exists; empty masks on no-tumor) + auxiliary deep supervision. AdamW, cosine schedule, EMA weights, bf16 AMP. 3 seeds; deployed seed 0, chosen on validation (best epoch 26). |
| Configuration choice | Pre-registered ablation on validation only: output stride {16, 8} × ε {0.005, 0.02, 0.05} (FINDINGS §10) |
| Deployment | ONNX opset 17, static INT8 (QDQ). The attention-pooling path and classifier stay FP32, so the exact CAM identity holds on the device (decomposition error ~1e-7). Size 96.6 → 24.7 MB. |
| Input | 224×224 after a foreground (brain) crop, grayscale ×3, normalized. PNG, JPG or DICOM; DICOM PHI is stripped at decode; images above 16 MP are refused. |
| Outputs | Calibrated probabilities, class-conditional 95% conformal set, OOD verdict (warn / reject), exact Grad-CAM, lesion mask with position and area, review flags |

## Data
* **Splits (v2):** BRISC 2025 (T1 contrast-enhanced, expert masks), pooled and re-split by patient-level groups:
  * train 3,766, used for training;
  * val 765, used for model selection;
  * **cal 670, used only for temperature, conformal, OOD and review thresholds**;
  * test 799, identical to the v0 test.
  * No group crosses splits. The official BRISC split leaks: 566 of its 1,000 test images are copies or mirrors
    of training images (FINDINGS §1).
* **External test:** BDNeuro-MRI, deduplicated against BRISC: 1,674 of 5,941 images remain after copy removal.

## Performance (INT8 as deployed)
| Set | n | Accuracy | Macro-F1 [95% CI] | 3-seed macro-F1 | v0 INT8 |
|---|---|---|---|---|---|
| Test (grouped, held out) | 799 | 0.949 | **0.944** [0.926, 0.959] | 0.938 ± 0.007 | 0.935 |
| **External BDNeuro (deduplicated)** | 1,674 | 0.866 | **0.836** [0.815, 0.856] | 0.827 ± 0.012 | 0.729 |

* **Paired test vs v0:** ΔF1 = +0.009 [−0.005, +0.025]; McNemar p = 0.23. v1 is non-inferior to v0, not
  proven better.
* **FP32 vs INT8:** external macro-F1 0.842 → 0.836. v0 lost 0.843 → 0.729 here, which its docs had reported as
  0.843.
* **Pi 5 vs PC** on 40 scans: 40/40 identical predictions; max |Δp| 0.010; CAM Pearson ≥ 0.997.
* **Pi 5 speed:** full analysis 238 ms median, 250 ms p95.
* **Calibration:** temperature T = 2.56, fitted on cal; test ECE 0.042 → 0.024.
* **Conformal 95% sets** cover **0.911** on test. That is up from 0.867, but **below the 0.93 target**.
  * Per class: glioma 0.83, meningioma 0.93, no tumor 0.99, pituitary 0.89.
  * Coverage is 0.96 on cal, so the test release is not exchangeable with the cal images.
  * Treat the sets as approximate; an empty set is flagged for review.
* **OOD:** multi-layer Mahalanobis, fitted on the deployed graph. 6.3% of test images are flagged "unusual".

## Explanations
* **What is shown:** Grad-CAM and the lesion outline, nothing else. The Grad-CAM is the exact additive
  decomposition of the logit (HiResCAM form), displayed with a monotone colour map.
* **Measured on 120 correctly classified test tumors:**

  | Measure | v1 | v0 |
  |---|---|---|
  | Evidence inside the tumor | **0.73**, 98% of the 28×28 oracle ceiling | 0.15 |
  | Pointing game | **0.99** | 0.53 |
  | ROAD | **0.32** | 0.29 |

* **Per class:** glioma 0.64, meningioma 0.84, pituitary 0.70. v0's pituitary maps highlighted the surrounding
  anatomy instead of the tumor.
* **Caveat:** the map's spatial envelope comes partly from the segmentation-derived attention.
  * After the classifier is randomized, the rank correlation is destroyed (Spearman −0.04), but SSIM stays 0.81.
  * Read the map as **lesion-weighted classifier evidence**.

## Clinical portal (edge/portal)
* **Roles:** patient, doctor and admin.
* **Workflow:** the patient uploads; the AI result goes to a prioritized doctor worklist; the doctor signs the
  report; the patient sees the doctor's report only.
* **Security:**
  * argon2 passwords, lockout and rate limits, CSRF, strict CSP;
  * object-level access checks and an audit log;
  * scans, overlays and signed reports encrypted at rest (AES-256-GCM);
  * reports are versioned and never overwritten.
* **Doctor assistant:** Qwen3-1.7B via llama.cpp, grounded in the case facts and a cited knowledge base. Every
  answer passes a deterministic verifier, and statistical questions get fixed template answers.
* **Consultations:** after the report, the patient books a follow-up with the reporting radiologist. The
  radiologist records attendance. Reminders are delivered in-app and by e-mail, and the e-mails never contain
  clinical content.
* **Follow-up risk score (CareLoop):** this is not a learned model. It is a fixed, documented weighted sum of
  operational signals, used only to order the coordinator's follow-up worklist:
  * days overdue (1.5 points per day, at most 35), or due within the "due soon" window with nothing booked (6);
  * overdue with no contact attempt yet (5);
  * missed appointments (15 each, at most 30);
  * unanswered contact attempts (8 each, at most 24);
  * contact details not working (15);
  * a referral not completed (15);
  * travel distance over 20 km (5) or over 50 km (10);
  * an appointment booked (−20).

  Bands: High is 60 or more, Medium 30 or more. It never uses the diagnosis, the report or any AI output, so it
  cannot encode clinical bias from the classifier. It decides nothing: every follow-up plan is approved by a
  doctor and every reminder is sent by a person. A learned model (for example gradient boosting) would need real
  follow-up outcomes from a deployment, and is future work.

## Intended use and out-of-scope use
* **Intended:** research, teaching and demonstrations of explainable, secure edge AI on public data.
* **Out of scope:**
  * clinical use;
  * CT and non-T1-weighted sequences;
  * children and post-operative scans;
  * tumor types outside the four classes (e.g. metastases);
  * full 3D studies.

## Known limitations and risks
* **Sequence shortcut:** BRISC's "no tumor" class mixes CT, FLAIR, T2 and non-contrast T1, while all tumor
  classes are T1 contrast-enhanced. A fix is planned with BraTS data.
* **Coverage below target** on the test release (see above).
* **Single 2D slice:** a "no tumor" output cannot exclude a lesion elsewhere in the brain. Upload the whole series
  instead (below).
* **Whole-study mode** (`Engine.analyze_series`; NIfTI, a .zip of a series, or all DICOM files of one series):
  36 slices in 3 planes, combined with the top-5-slices rule (chosen on calibration patients). On held-out patients
  (BraTS-GLI 76, BraTS-MEN 68, NHNN pituitary 15, IXI healthy 380):
  * this model (`mri/0.2.0`): tumour-type macro-F1 0.71, slice AUROC 0.72, healthy false-alarm rate 23%. **It misses
    all three pre-registered targets; do not use whole-study mode with it.**
  * the retrained candidate (`mri_v2vol`, seed 0; not yet released): macro-F1 0.98, slice AUROC 0.95, false-alarm rate
    5% (all targets met). These test patients come from the same datasets as its training patients; the external
    BraTS-Africa test and two more seeds are still to come.
  * The 95% confidence set is calibrated on slices, so its coverage at patient level is not guaranteed.
* **Image orientation:** PNG/JPG files carry no orientation, so image left/right is not the patient's left/right.
* **Distribution shift:** accuracy drops on other hospitals' data (external macro-F1 0.84). Pituitary is the
  weakest external class.

## Security and privacy
* **Model distribution:**
  * signed with OpenSSF Model Signing (EC P-256, offline key);
  * encrypted with AES-256-GCM;
  * the data key is wrapped per device with X25519.
* **On the Pi:**
  * the model is decrypted and verified in RAM only;
  * the portal is served over HTTPS only (Caddy, local CA), and the app listens on loopback;
  * an nftables firewall admits only SSH, HTTP(S) and mDNS;
  * patient data is encrypted at rest with a separate key.
* **Threat model limit:** someone with root access on an enrolled device can read the model and the portal key.
  The Pi 5 has no hardware key store.

Full evidence and methods: [`docs/FINDINGS.md`](FINDINGS.md).
