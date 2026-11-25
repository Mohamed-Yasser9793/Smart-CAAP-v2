# Smart CAAP v2.0

**Quality-Guided Adaptive Radiograph Routing and Post-Hoc Confidence Calibration for Frozen Musculoskeletal Fracture Classifiers**

Smart CAAP v2.0 is a lightweight, training-free preprocessing and calibration gateway placed around a frozen ResNet-50 wrist fracture classifier (`weights/best_resnet50_wrist.pth`). It inspects the central 60% of each wrist X-ray, skips clean scans untouched, restores dark or noisy scans without amplifying noise, and calibrates study-level fracture probabilities.

---

## How It Works

1. **Central 60% ROI Quality Check (`caap/pipeline.py`):** Measures contrast span ($P_{98} - P_2$) and Laplacian noise energy inside the central 60% of the radiograph (ignoring dark borders and `L`/`R` markers).
2. **3-Path Adaptive Router:**
   - **Path 1 — Clean Bypass:** Good contrast & low noise $\rightarrow$ image unchanged ($\mathrm{SSIM} = 1.0000$).
   - **Path 2 — ROI Soft Stretch:** Low contrast & low noise $\rightarrow$ ROI-anchored contrast stretch blended at $\alpha = 0.75$.
   - **Path 3 — Bilateral Denoise + Stretch:** High noise $\rightarrow$ edge-preserving bilateral filter ($d=5, \sigma=25$) + ROI stretch blended at $\alpha = 0.80$.
3. **Frozen ResNet-50 + Calibration (`caap/inference.py`):** Pools mean view logits per study and applies validation-fitted post-hoc temperature scaling ($T^*$). No model weights are retrained.

---

## Project Structure

- `app.py` — Interactive Streamlit workstation (live X-ray analysis, Grad-CAM, and audited results dashboard)
- `caap/` — Core Smart CAAP v2.0 router, frozen ResNet-50 inference, and charts
- `experiments/` — Standalone CLAIM 2024 evaluation & Grad-CAM failure-analysis scripts
- `assets/claim_audit/` — Audited Split B CSVs, LaTeX tables, SHA-256 manifest, and 10 Grad-CAM figures
- `weights/best_resnet50_wrist.pth` — Frozen ResNet-50 checkpoint (tracked with Git LFS)
- `notebooks/` — Original archival research notebook

---

## Quick Start

### 1. Install Dependencies (Python 3.11+)
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the Interactive Dashboard
```bash
streamlit run app.py
```

### 3. Reproduce the Paper Results (Requires Stanford MURA v1.1)
Stanford MURA v1.1 is **not** included due to its data-use agreement. Download it from the [Stanford MURA portal](https://stanfordmlgroup.github.io/competitions/mura/) (or mount it on Kaggle at `/kaggle/input`), then run:

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

---

## Key Audited Results (Split B Test, n = 738 Studies, Seed = 42)

- **Patient-Disjoint Split B:** 3,474 patients (2,588 train / 371 val / 738 test studies; zero patient overlap).
- **Full Test (`n = 738`):** Highest sensitivity (**87.59%**) and NPV (**91.67%**); ROC-AUC **0.9434** (+0.0147 vs. Blind Stretch, DeLong $p = 0.00105$).
- **Natural Degraded (`n = 518`):** Sensitivity **90.45%**, NPV **92.28%** (21 missed fractures vs. 50 for CLAHE).
- **Moderate Stress (Tier 2):** ROC-AUC **0.7430** and **91.72%** sensitivity (vs. Raw 0.6352 / 20.69%, $p = 4.24 \times 10^{-8}$).
- **Severe Noise (Tier 3):** Prevents specificity collapse (**73.21%** vs. 0.00% CLAHE and 1.56% Blind Stretch) with ECE **0.0849** ($p < 0.0001$).
- **Speed:** `0.9772 ms/image` single-thread CPU (zero GPU overhead).

---

## Limitations & Disclaimer

- Evaluated internally on Stanford MURA v1.1 `XR_WRIST` with a single frozen ResNet-50 backbone; external multi-hospital validation was not performed.
- MURA v1.1 does not contain demographic metadata (age, sex, ethnicity).
- **Research prototype only** — not approved for clinical diagnosis or triage.

---

## Citation

```bibtex
@inproceedings{shehata2026smartcaap,
  title  = {Smart CAAP v2.0: Quality-Guided Adaptive Radiograph Routing and Post-Hoc Confidence Calibration for Frozen Musculoskeletal Fracture Classifiers},
  author = {Shehata, Mohamed Yasser},
  year   = {2026}
}
```
