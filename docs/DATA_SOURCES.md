# 3D datasets for patient-level evaluation (FINDINGS §11)

A radiologist decides from the whole study, not one image, so the model is also evaluated on complete 3D scans. All
data stays outside the repository, in `~/neuroedge_data/raw/<folder>/`. Only the post-contrast T1 (T1c) volume (T1 for
IXI) and the tumour mask are kept for each patient.

| Folder | Dataset | Class | Patients used | Role | Access / terms | Citation |
|---|---|---|---|---|---|---|
| `brats_gli` | BraTS 2023 adult glioma | glioma | about 300 (subset) | development (calibration / test patients; training if retrained) | Synapse account; accept the BraTS terms (research use) | Baid et al. 2021; Menze et al. 2015 |
| `brats_africa` | BraTS-Africa 2023 | glioma | all (count recorded on download) | **external test** | Synapse account; accept the terms | Adewole et al. 2023 |
| `brats_men` | BraTS-MEN 2023 (pre-operative meningioma) | meningioma | about 300 (subset) | development | Synapse account; accept the terms | LaBella et al. 2024, *Sci Data* |
| `tcia_meningioma` | TCIA Meningioma-SEG-CLASS | meningioma | 96 | **external test** | TCIA Data Retriever; CC BY 4.0; NIH controlled-access policy for face-reconstructable images | Vassantachart et al., TCIA DOI 10.7937/0TKV-1A36 |
| `pituitary_nhnn` | NHNN pituitary tumour MRI (T1c, tumour + carotid masks) | pituitary | 60 of 136 (patients 1–60) | development (a held-out part is the test) | Open; CC BY (figshare 10.6084/m9.figshare.27894084) | Pandit et al. 2025, *Sci Data*, DOI 10.1038/s41597-024-04218-8 |
| `ixi` | IXI healthy volunteers (T1, **no contrast**) | no tumour | about 580 | **external test** (reported separately) | Open download; CC BY-SA 3.0 | brain-development.org/ixi-dataset |

**Not used:** UCSF-PDGM (TCIA). Its archive is very large, and BraTS already contains UCSF patients, so using it would
risk testing on patients the model trained on.

## Leakage control
* **Within a dataset:** splits are by patient (`study_id`), so all slices of one patient are on one side.
* **Across datasets:** `python -m training.data.check_overlap` compares every pair of studies from different datasets,
  and every study against the old 2D training images (BRISC / figshare / Nickparvar). It uses the same flip-aware
  thumbnail similarity and thresholds as the 2D deduplication (`t_dup` 0.98, `t_patient` 0.94). Matched patients are
  excluded from all test sets; the counts are in `manifests/volumes_overlap.json` and FINDINGS §11.

## Known caveats
* **Preprocessing shift:** BraTS volumes are skull-stripped and resampled to 1 mm, while BRISC (training) and the
  pituitary set keep the skull. Results are therefore reported per dataset, so a dataset shortcut cannot hide in a
  pooled number.
* **IXI has no contrast agent:** a "no tumour" result on IXI is a weaker test than on contrast scans.
* **Tumour-free slices:** slices of the tumour patients without any tumour (from the BraTS / pituitary masks) give a
  same-scanner "no tumour" test.

## Steps
1. **IXI:** download `IXI-T1.tar`, then run `python -m training.data.download_3d ixi`.
2. **BraTS:** create a Synapse account, accept the terms for each dataset, run `synapse login`, then for each dataset
   run `python -m training.data.download_3d synapse --id <syn id from the dataset page> --dest brats_gli --limit 300`.
3. **TCIA meningioma:** download it with the TCIA Data Retriever (T1c + segmentation series) into `raw/tcia_meningioma/`.
4. **Pituitary:** `python -m training.data.remote_zip fetch https://ndownloader.figshare.com/files/50793273 --dest ~/neuroedge_data/raw/pituitary_nhnn/data --match "^([1-9]|[1-5][0-9]|60)_(T1|tumor_f).nii.gz$"` (fetches only those files out of the 47 GB zip).
5. **Prepare and run:**
   * `python -m training.data.build_volumes`
   * `python -m training.data.check_overlap`
   * `python -m training.evaluate_volumes`
