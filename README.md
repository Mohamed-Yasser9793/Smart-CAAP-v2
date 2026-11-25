# Smart CAAP v2.0: Quality-Guided Adaptive Radiograph Routing and Post-Hoc Confidence Calibration for Frozen Musculoskeletal Fracture Classifiers

## 1. Research Objective
**Smart CAAP v2.0** (*Contrast- and Constraint-Aware Adaptive Pipeline*) is a training-free, model-agnostic radiograph preprocessing and post-hoc confidence calibration gateway designed to operate strictly around a frozen musculoskeletal fracture classifier (`best_resnet50_wrist.pth`, ResNet-50) without modifying or retraining backbone weights. This repository provides the complete modular implementation, interactive Streamlit research workstation, and standalone **IEEE / CLAIM 2024** reproducibility and statistical audit scripts.

---

## 2. Clinical and Research Motivation
In emergency department and portable bedside wrist radiography, suboptimal tube exposure, soft-tissue attenuation, patient motion, and detector variability frequently introduce contrast compression (fading), high-frequency quantum sensor noise, dark collimation borders, and peripheral radiopaque lead laterality markers (`L`/`R`). Prepending unconditional static enhancement filters—such as **CLAHE** or **global percentile contrast stretching**—in front of a frozen classifier trained on raw radiographs causes two failure modes:
1. **Unconditional Manifold Drift:** Remapping already clean radiographs shifts pixel statistics away from the frozen network's training distribution.
2. **Quadratic Noise Amplification:** Global stretching on underexposed, noisy scans amplifies additive noise variance inversely with the squared compressed dynamic range, triggering false-positive fracture activations and specificity collapse.

Smart CAAP v2.0 resolves these hazards by gating enhancement behind central-ROI quality sensors, preserving clean radiographs via a zero-distortion bypass, bounding pixel excursion with residual alpha-blending, and applying symmetric validation-fitted post-hoc temperature scaling.

---

## 3. High-Level Smart CAAP v2.0 Architecture
Smart CAAP v2.0 executes three stages around the frozen ResNet-50 classifier (`320 × 320` grayscale input, `INTER_AREA` resize):

1. **Stage 1 — Central 60% Anatomical ROI Quality Sensing (`caap/pipeline.py`):** Crops the central 60% skeletal window (`192 × 192` px, excluding outer 20% collimation borders and `L`/`R` markers) and computes:
   - Robust dynamic contrast span: $C_{\text{span}} = P_{98}(\text{ROI}) - P_{2}(\text{ROI})$
   - High-frequency Laplacian noise energy: $E_{\text{HF}} = \mathrm{Var}(\nabla^2 \text{ROI})$
   - Validation-locked routing thresholds calibrated strictly on Split B Validation ($n = 371$ studies, zero test leakage): $\tau_c = 0.2353$ (35th percentile) and $\tau_n = 0.003342$ (75th percentile).
2. **Stage 2 — Tri-Path Conditional Routing & Residual Blending:**
   - **Path 1 (Clean Bypass, $\alpha_1 = 0.0$):** $C_{\text{span}} \ge \tau_c \land E_{\text{HF}} \le \tau_n \rightarrow$ exact identity pass-through ($\mathrm{SSIM} = 1.0000$, $\mathrm{PSNR} = \infty$).
   - **Path 2 (ROI-Anchored Soft Stretch, $\alpha_2 = 0.75$):** $C_{\text{span}} < \tau_c \land E_{\text{HF}} \le \tau_n \rightarrow$ linear stretch anchored to central ROI $[P_2, P_{98}]$ (global $[P_1, P_{99}]$ fallback if $P_{98} - P_2 < 10$ gray levels), softly blended with raw input.
   - **Path 3 (Edge-Preserving Bilateral Denoise + Soft Stretch, $\alpha_3 = 0.80$):** $E_{\text{HF}} > \tau_n \rightarrow$ bilateral filter ($d=5, \sigma_{\text{color}}=25, \sigma_{\text{space}}=25$) followed by ROI-anchored soft stretch.
3. **Stage 3 — Frozen ResNet-50 Study-Level Pooling & Symmetric Calibration (`caap/inference.py`, `experiments/metrics_stats.py`):** Arithmetic mean of view logits per study followed by validation-fitted `L-BFGS-B` post-hoc temperature scaling ($T^*$: Raw `0.9658`, CLAHE `1.0558`, Blind Stretch `1.2405`, Smart CAAP v2.0 `0.9460`).

**CPU Preprocessing Latency:** `0.9772 ms/radiograph` single-thread end-to-end router latency (Intel Xeon Platinum 8481C, 1,000 iterations, `cv2.setNumThreads(1)`), complementing the original `~0.11 ms/image` core-kernel benchmark, with zero GPU overhead.

---

## 4. Dataset, Data-Access Restrictions & Redistributability
| Asset Category | Included in This Package? | Access / Governance Policy |
| :--- | :---: | :--- |
| **Source Code (`app.py`, `caap/`, `experiments/`)** | **Yes** | Publicly redistributable research code. |
| **Audited Tables, Manifests & Grad-CAM Panels (`assets/claim_audit/`)** | **Yes** | De-identified aggregate metrics, SHA-256 split hashes, LaTeX tables, and 10 audited failure-case visualizations. |
| **Frozen ResNet-50 Weights (`weights/best_resnet50_wrist.pth`)** | **Yes** (~94.4 MB) | Track via **Git LFS** on GitHub (`git lfs track "weights/*.pth"`). Trained on MURA v1.1 `XR_WRIST` for academic research only. |
| **Stanford MURA v1.1 Raw Radiographs** | **No (Restricted)** | **Not redistributed.** Researchers must obtain MURA v1.1 directly from the Stanford Machine Learning Group under its official data-use agreement (`https://stanfordmlgroup.github.io/competitions/mura/`). |

---

## 5. Split B Patient-Level Separation (Zero Leakage)
To eliminate patient identity leakage across longitudinal studies, `experiments/data_splits.py` implements nested `StratifiedGroupKFold` (`random_state=42`, 5-fold outer then 8-fold inner) enforcing $\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} \cap \mathcal{P}_{\text{test}} = \emptyset$ (verified programmatically and locked by SHA-256):
- **Total MURA `XR_WRIST`:** 3,474 unique patients | 3,697 studies | 10,411 views (2.82 ± 0.84 views/study; 38.5% positive; 41.7% hardware-flagged)
- **Train (Split B):** 2,434 patients | 2,588 studies | 7,257 views
- **Validation (Split B):** 347 patients | 371 studies | 1,063 views
- **Test — Full (Split B):** 693 patients | 738 studies | 2,091 views (SHA-256: `eef7f9a849304cf84db9c3c6a2fcac0ad579512aafaf0df6fc06c262aa648dfc`)
- **Natural Degraded Test Cohort:** 518 studies (220 positive, 298 negative) triggering Path 2 or Path 3.

---

## 6. Repository Structure
```text
Smart-CAAP-v2/
├── README.md                                  # Project & CLAIM 2024 reproducibility documentation
├── requirements.txt                           # Pinned Python 3.11+ dependencies
├── .gitignore                                 # Ignore rules + Git LFS guidance for *.pth
├── app.py                                     # Interactive Streamlit research workstation
├── caap/                                      # Core Smart CAAP v2.0 runtime
│   ├── __init__.py
│   ├── pipeline.py                            # Central 60% ROI sensors, tri-path router, hardware proxy
│   ├── inference.py                           # Frozen ResNet-50 loader, symmetric T*, live Grad-CAM
│   ├── audit_loader.py                        # Loader for bundled CLAIM 2024 audit outputs
│   ├── charts.py                              # Plotly charts (95% CI bars, confusion matrices, stress curves)
│   ├── records.py                             # Original notebook benchmark records
│   └── theme.py                               # Workstation styling
├── experiments/                               # Standalone CLAIM 2024 reproducibility & audit suite
│   ├── __init__.py
│   ├── data_splits.py                         # Exact Split A & Split B builders + CLAIM Items 35-36 table
│   ├── metrics_stats.py                       # Symmetric T*, ECE/Brier/NLL/RPI, DeLong, McNemar, Bootstrap
│   ├── run_claim_audit_extensions.py          # One-click runner for Tables I, II, V & statistical audits
│   └── gradcam_failure_analysis.py            # ResNet-50 layer4 Grad-CAM FP/FN visualizer (CLAIM 31 & 39)
├── assets/claim_audit/                        # Bundled verified Split B CSVs, JSON manifest, .tex & 10 Grad-CAM PNGs
├── notebooks/
│   └── smart_caap_v2_experiments.ipynb        # Archival experimental notebook
├── weights/
│   └── best_resnet50_wrist.pth                # Frozen ResNet-50 checkpoint (Git LFS)
└── outputs/
    └── .gitkeep                               # Output folder for fresh local re-runs
```

---

## 7. Environment & Dependencies (Python 3.11+)
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
Key pinned versions matching the paper: `numpy==2.1.3`, `opencv-python-headless==4.11.0.86`, `scipy==1.15.2`, `scikit-learn==1.6.1`, `scikit-image==0.25.2`, `pandas==2.2.3`, `matplotlib==3.10.0`, `torch>=2.5.1,<2.7.0`, `torchvision>=0.20.1,<0.22.0`, `streamlit>=1.42.0,<2.0.0`.

If cloning from GitHub after pushing with Git LFS:
```bash
git lfs install
git lfs pull
```

---

## 8. Running the Pipeline & Reproducing All Reported Evaluations
### A. Reproduce All Audited Tables, CIs, DeLong/McNemar Tests & Grad-CAM Panels (`seed=42`)
Download Stanford MURA v1.1 to `<MURA_ROOT>` (or mount at `/kaggle/input`) and run:
```bash
python -m experiments.run_claim_audit_extensions \
    --mura-root /kaggle/input \
    --weights weights/best_resnet50_wrist.pth \
    --output-dir outputs/claim_audit \
    --n-boot 1000 \
    --seed 42 \
    --stress-seeds 42 \
    --run-gradcam
```
This deterministic command reproduces:
- `dataset_partition_summary.csv` & `.tex` (CLAIM Items 35–36)
- `split_b_tables_1_2_metrics_cis.csv` (Full Test $n=738$ & Natural Degraded $n=518$, Uncalibrated & Symmetric $T^*$, 95% bootstrap CIs, exact `TP/TN/FP/FN`)
- `split_b_table_5_seeded_stress_test.csv` (4-tier deterministic synthetic stress test, `seed=42`)
- `split_b_significance_delong_mcnemar_boot.csv` (exact DeLong $z/p$, McNemar $b, c, p$, paired bootstrap $\Delta$ CIs)
- `ieee_latex_tables.tex`, `audit_manifest.json`, and `gradcam_failures/` (10 300-DPI FN/FP Grad-CAM panels)

### B. Standalone Grad-CAM Failure Analysis Only
```bash
python -m experiments.gradcam_failure_analysis \
    --mura-root /kaggle/input \
    --weights weights/best_resnet50_wrist.pth \
    --output-dir outputs/gradcam_failures \
    --top-k 5
```

### C. Launch the Interactive Streamlit Research Workstation
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`, automatically loading `weights/best_resnet50_wrist.pth` and the bundled audit snapshot in `assets/claim_audit/` (live radiograph inference, live ResNet-50 `layer4` Grad-CAM, symmetric $T^*$ toggle, 95% CI charts, confusion matrices, and the 10-case failure gallery).

---

## 9. Important Limitations & Clinical Scope
- **Single-Institution Scope & No External Validation:** All audited Split B results are derived strictly from the internal Stanford MURA v1.1 `XR_WRIST` cohort; external multi-hospital validation was not performed in this study.
- **Absent Demographic Metadata:** MURA v1.1 does not include patient age, sex, or ethnicity, precluding demographic fairness stratification.
- **Single Backbone & Native Raw Bias:** Evaluated on a single frozen ResNet-50 backbone trained on raw MURA radiographs.
- **Research-Only Prototype:** Smart CAAP v2.0 is a retrospective research preprocessing and calibration prototype and is **not** cleared or intended for autonomous clinical diagnosis or patient triage.

---

## 10. Citation & Archival Links
- **GitHub Repository:** `[GitHub Repository URL — to be inserted upon public release]`
- **Zenodo Archive:** `[Zenodo DOI — to be inserted upon archival release]`

```bibtex
@inproceedings{shehata2026smartcaap,
  title     = {Smart CAAP v2.0: Quality-Guided Adaptive Radiograph Routing and Post-Hoc Confidence Calibration for Frozen Musculoskeletal Fracture Classifiers},
  author    = {Shehata, Mohamed Yasser},
  year      = {2026},
  note      = {Code and reproducibility package (CLAIM 2024 audited)}
}
```

## 11. License
No software license file has been added to this repository yet. Add a `LICENSE` file (e.g., MIT or Apache-2.0 for code) prior to public GitHub/Zenodo publication while keeping Stanford MURA v1.1 data subject to the Stanford Machine Learning Group data-use agreement.
