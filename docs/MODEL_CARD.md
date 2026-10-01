# Model card: NeuroEdge MRI v0 (registry version `mri/0.1.1`)

> **Research and education only. Not a medical device.** Outputs must not be used for diagnosis or treatment
> decisions. Every result requires review by a qualified radiologist.

## Model
| | |
|---|---|
| Task | 4-class classification of a single 2D brain MRI slice (glioma, meningioma, no tumor, pituitary tumor) plus a tumor segmentation mask |
| Architecture | ResNet-50 (timm, ImageNet init), output stride 16 (14×14 feature map), **GAP → Dropout → Linear** head, FPN segmentation head |
| Training | Multi-task loss: cross-entropy + BCE + Dice, the segmentation terms only where a mask exists. AdamW, cosine schedule, EMA weights, bf16 AMP. Seed 0, best epoch 22 (validation macro-F1). |
| Deployment | ONNX opset 17 with static INT8 (QDQ) quantization. GAP and classifier kept in FP32 so the exact CAM identity holds on the device. Size 96.6 → 24.7 MB. |
| Input | 224×224 after a foreground (brain) crop, grayscale ×3, normalized. PNG, JPG or DICOM; DICOM PHI is stripped at decode. |
| Outputs | Calibrated probabilities, class-conditional 95% conformal prediction set, OOD verdict (warn / reject), exact Grad-CAM, Finer-CAM, LayerCAM (on demand), lesion mask with position and area, review flags |

## Data
* **Training, validation, test:** BRISC 2025 (T1 contrast-enhanced, expert masks), pooled and **re-split
  70/15/15 by patient-level groups**. The official split leaks: 566 of 1,000 official test images are copies or
  mirrors of training images. Groups use flip-aware image similarity plus true patient IDs carried over from
  figshare (`docs/FINDINGS.md` §1).
* **External test:** BDNeuro-MRI, deduplicated against BRISC; 1,674 of 5,941 images remain after copy removal.
  figshare and Nickparvar are almost entirely copies of BRISC, so their small residual sets are not independent
  evidence.

## Performance (FP32 unless noted)
| Set | n | Accuracy | Macro-F1 | Macro-AUROC |
|---|---|---|---|---|
| Test (grouped, held out) | 799 | 0.949 | 0.942 | 0.994 |
| Test, INT8 as deployed | 799 | 0.939 | 0.935 | 0.991 |
| **External BDNeuro (deduplicated)** | 1,674 | **0.864** | **0.843** | 0.968 |

* INT8 vs FP32: McNemar p = 0.38 (no significant difference); prediction agreement 95.7%.
* **Pi 5 vs PC** on 40 held-out scans: 40/40 identical predictions; max |Δp| = 0.016.
* Externally, pituitary is the weakest class (sensitivity 0.69).
* **Calibration:** temperature T = 1.575; test ECE 0.049 → 0.030.
* **Conformal 95% sets:** they **under-cover on test** (0.87 overall; glioma 0.59–0.73), because validation
  served both model selection and calibration. Treat them as approximate.
* **OOD:** multi-layer Mahalanobis with two tiers, fitted on the deployed INT8 graph. 7.4% of in-distribution
  images are flagged "unusual"; the reject threshold is the 99.5th percentile of validation scores.

## Explanations
* **Exact Grad-CAM.** With a GAP→Linear head, the map is an exact additive decomposition of the logit
  (decomposition error ≈ 3e-6 on INT8). No post-processing is applied: no thresholding, gamma or blob selection.
* **Finer-CAM** contrasts the prediction with the runner-up class.
* **LayerCAM** (layer3+4, FP32) is an optional, sharper detail view.
* **Measured on 120 test tumors:** 15% of the positive evidence lies inside the tumor (the 14×14 ceiling is
  55%); pointing game 0.53 (LayerCAM 0.76). Randomization sanity checks pass.
* **Pituitary shortcut:** pituitary maps highlight surrounding anatomy rather than the tumor. The separate
  segmentation head localizes tumors far better (84% of its energy inside the mask).

## Offline assistant
* Qwen3-1.7B (Q4_K_M) via llama.cpp, on the device.
* It writes narratives and answers questions from the model's facts and a cited knowledge base, and every
  answer passes a deterministic verifier.
* Questions about the prediction set, "why X", review status, treatment, or rejected inputs get fixed template
  answers; the LLM is not used for them.
* Fresh-case audit: 3 of 80 verified answers (3.8%) still contained a clear error. Human sign-off is mandatory.

## Intended use and out-of-scope use
* **Intended:** research, teaching and demonstrations of explainable, secure edge AI on public data.
* **Out of scope:** clinical use of any kind; CT; non-T1-weighted sequences; children; post-operative scans;
  tumor types outside the four classes (e.g. metastases); full 3D studies (a later phase).

## Known limitations and risks
* **Sequence shortcut:** BRISC's "no tumor" class mixes CT, FLAIR, T2 and non-contrast T1, while all tumor
  classes are T1 contrast-enhanced. The model can partly separate classes by sequence. A fix is planned with
  BraTS data.
* **Single 2D slice:** a "no tumor" output cannot exclude a lesion elsewhere in the brain.
* **Image orientation:** PNG/JPG files carry no orientation, so image left/right is not the patient's left/right.
* **Distribution shift:** accuracy drops on other hospitals' data (external macro-F1 0.84).
* **Single seed:** results are from one training seed; 3-seed mean ± sd is pending.

## Security and privacy
* **Model distribution:** signed with OpenSSF Model Signing (EC P-256, offline key) and encrypted with
  AES-256-GCM. The data key is wrapped per device with X25519 and HKDF.
* **On the device:** decrypted and verified in RAM only; the device stores ciphertext only.
* **Uploaded images:** processed in memory and expire after one hour; nothing patient-related is written to disk.
* **Threat model limit:** someone with root access on an enrolled device can extract the model from RAM.

Full evidence and methods: [`docs/FINDINGS.md`](FINDINGS.md).
