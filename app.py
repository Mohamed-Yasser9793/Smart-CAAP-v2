"""Smart CAAP v2.0 Research Workstation & CLAIM 2024 Audit Dashboard.

Interface only. Signal routing lives in caap.pipeline, model inference & live
Grad-CAM live in caap.inference, bundled CLAIM 2024 audit results live in
caap.audit_loader (assets/claim_audit/), and original notebook records live in
caap.records.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from caap.audit_loader import (
    audit_available,
    load_gradcam_cases,
    load_manifest,
    load_partition_summary,
    load_significance,
    load_table_5,
    load_tables_1_2,
    load_text_file,
)
from caap.charts import (
    confusion_matrix_heatmap,
    intensity_histogram,
    metric_bars,
    metric_bars_with_ci,
    probability_bars,
    routing_donut,
    tier_lines,
    training_figure,
)
from caap.inference import SYMMETRIC_TEMPS, WEIGHTS_PATH, analyze_radiograph, load_backbone
from caap.pipeline import (
    OPTIMAL_T_CAL,
    PATH_BYPASS,
    PATH_DENOISE,
    PATH_STRETCH,
    TAU_CONTRAST_CAL,
    TAU_NOISE_CAL,
    THRESHOLD_STANDARD,
    YOUDEN_T3,
    prepare_radiograph,
    simulate_degradation,
    synthetic_phantom,
    wipe_compare,
)
from caap.records import (
    BEST_VAL_AUC,
    EARLIER_ROUTER,
    EARLIER_ROUTER_NOTE,
    FULL_TEST,
    FULL_TEST_NOTE,
    NATURAL_COHORT,
    NATURAL_COHORT_NOTE,
    ROUTING_FULL_TEST,
    ROUTING_NATURAL,
    SPLIT_COUNTS,
    SYNTHETIC_NOTE,
    SYNTHETIC_TIERS,
    TRAINING_LOG,
)
from caap.theme import inject, note, page_header

st.set_page_config(
    page_title="Smart CAAP v2.0 Research Workstation",
    page_icon=":material/radiology:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject()

CI_MAP = {
    "ROC-AUC": ("AUC_CI_L", "AUC_CI_H"),
    "ECE": ("ECE_CI_L", "ECE_CI_H"),
    "Brier": ("Brier_CI_L", "Brier_CI_H"),
    "NLL": ("NLL_CI_L", "NLL_CI_H"),
    "RPI (%)": ("RPI_CI_L", "RPI_CI_H"),
    "Sensitivity": ("Sens_CI_L", "Sens_CI_H"),
    "Specificity": ("Spec_CI_L", "Spec_CI_H"),
    "PPV": ("PPV_CI_L", "PPV_CI_H"),
    "NPV": ("NPV_CI_L", "NPV_CI_H"),
}


@st.cache_resource(show_spinner="Loading ResNet-50 checkpoint")
def get_model():
    return load_backbone(WEIGHTS_PATH)


@st.cache_data
def get_audit_bundle() -> dict:
    return {
        "available": audit_available(),
        "manifest": load_manifest(),
        "partitions": load_partition_summary(),
        "tables_1_2": load_tables_1_2(),
        "table_5": load_table_5(),
        "significance": load_significance(),
        "gradcam": load_gradcam_cases(),
        "latex_tables": load_text_file("ieee_latex_tables.tex"),
        "latex_partitions": load_text_file("dataset_partition_summary.tex"),
        "manifest_raw": load_text_file("audit_manifest.json"),
    }


def _read_upload(uploaded) -> np.ndarray:
    try:
        image = Image.open(BytesIO(uploaded.getvalue()))
    except UnidentifiedImageError as exc:
        raise ValueError("The file is not a readable PNG or JPEG image.") from exc
    return prepare_radiograph(image)


def _sidebar(model_meta: dict | None, model_error: str | None) -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-block">
              <div class="mark">Musculoskeletal AI · CLAIM 2024</div>
              <h2>Smart CAAP v2.0</h2>
              <p>Quality-guided routing & post-hoc calibration for a frozen wrist ResNet-50.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if model_error:
            st.error(model_error)
        elif model_meta:
            st.caption(
                f"Checkpoint `{model_meta['filename']}` · {model_meta['parameters']:,} params · {model_meta['device']}"
            )

        st.markdown("##### Radiograph")
        uploaded = st.file_uploader(
            "Upload a wrist radiograph",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            help="Grayscale conversion and a 320×320 area resize match training.",
        )
        load_phantom = st.button("Load synthetic phantom", width="stretch")
        use_upload = st.button("Use uploaded radiograph", width="stretch", disabled=uploaded is None)

        digest = hashlib.sha256(uploaded.getvalue()).hexdigest() if uploaded is not None else None
        if digest and digest != st.session_state.get("active_digest"):
            st.session_state["source_u8"] = _read_upload(uploaded)
            st.session_state["image_name"] = uploaded.name
            st.session_state["synthetic"] = False
            st.session_state["active_digest"] = digest
        if load_phantom:
            st.session_state["source_u8"] = synthetic_phantom()
            st.session_state["image_name"] = "synthetic_phantom.png"
            st.session_state["synthetic"] = True
        if use_upload and uploaded is not None:
            st.session_state["source_u8"] = _read_upload(uploaded)
            st.session_state["image_name"] = uploaded.name
            st.session_state["synthetic"] = False
            st.session_state["active_digest"] = digest

        if st.session_state.get("source_u8") is not None:
            kind = "Synthetic phantom" if st.session_state.get("synthetic") else "Uploaded radiograph"
            st.caption(f"{kind}: {st.session_state.get('image_name')}")

        st.markdown("##### Calibration & operating point")
        cal_mode = st.radio(
            "Temperature scaling",
            options=[
                "Symmetric Split-B Val T* (CLAIM 2024)",
                "Original notebook T (CAAP=1.2215)",
            ],
            help="Symmetric T* applies validation-fitted L-BFGS-B temperatures to all four pipelines.",
        )
        st.session_state["symmetric_calibration"] = "Symmetric" in cal_mode

        choice = st.radio(
            "Decision threshold",
            options=[
                f"Standard  ·  {THRESHOLD_STANDARD:.2f}",
                f"Recorded Youden  ·  {YOUDEN_T3:.3f}",
            ],
            help="Full-test tables use 0.50. YOUDEN_T3 = 0.330 is the stored ER-triage threshold.",
        )
        st.session_state["threshold"] = YOUDEN_T3 if "Youden" in choice else THRESHOLD_STANDARD

        st.session_state["live_gradcam"] = st.toggle(
            "Compute live ResNet-50 Grad-CAM",
            value=st.session_state.get("live_gradcam", False),
            help="Runs one backward pass through ResNet-50 layer4[-1] to visualize spatial attention.",
        )

        st.markdown("##### Linear degradation")
        st.session_state["apply_degradation"] = st.toggle(
            "Compress contrast before inference",
            value=st.session_state.get("apply_degradation", False),
            help="Mean-centered contrast compression and seeded Gaussian noise (seed=42).",
        )
        st.session_state["alpha"] = st.slider("Contrast factor alpha", 0.20, 1.00, 1.00, 0.05)
        st.session_state["sigma"] = st.slider("Sensor noise sigma", 0.0, 10.0, 0.0, 0.5)


def _model_input(source: np.ndarray) -> np.ndarray:
    if not st.session_state.get("apply_degradation"):
        return source
    return simulate_degradation(
        source,
        alpha=float(st.session_state.get("alpha", 1.0)),
        sigma=float(st.session_state.get("sigma", 0.0)),
    )


def current_analysis(model):
    source = st.session_state.get("source_u8")
    if source is None or model is None:
        return None
    model_input = _model_input(source)
    threshold = float(st.session_state.get("threshold", THRESHOLD_STANDARD))
    sym_cal = bool(st.session_state.get("symmetric_calibration", True))
    with_gradcam = bool(st.session_state.get("live_gradcam", False))
    key = (
        source.tobytes(),
        model_input.tobytes(),
        round(threshold, 4),
        sym_cal,
        with_gradcam,
        bool(st.session_state.get("synthetic")),
    )
    if st.session_state.get("analysis_key") != key:
        with st.spinner("Running frozen ResNet-50 across all four pipelines"):
            result = analyze_radiograph(
                model,
                model_input,
                threshold,
                symmetric_calibration=sym_cal,
                with_gradcam=with_gradcam,
            )
        result["source"] = source
        result["image_name"] = st.session_state.get("image_name", "radiograph.png")
        result["synthetic"] = bool(st.session_state.get("synthetic"))
        st.session_state["analysis"] = result
        st.session_state["analysis_key"] = key
    return st.session_state["analysis"]


def _score_panel(result: dict) -> None:
    caap = result["caap_result"]
    probability = caap["probability"]
    reached = probability >= result["threshold"]
    status = "At or above the operating point" if reached else "Below the operating point"
    cal_tag = "Symmetric Split-B T*" if result.get("symmetric_calibration") else "Notebook T"
    st.markdown(
        f"""
        <div class="score-panel">
          <div class="score-kicker">Smart CAAP v2.0 · {cal_tag} = {caap['temperature']:.4f}</div>
          <div class="score-value">{probability:.1%}</div>
          <div class="score-label">Positive-class probability from the frozen checkpoint</div>
          <div class="score-sub">
            Logit {caap['logit']:.4f} · raw sigmoid {caap['raw_probability']:.4f}<br>
            Operating point {result['threshold']:.3f} · {status}
          </div>
          <div class="pill">{result['route']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _image_block(title: str, image: np.ndarray, caption: str) -> None:
    st.markdown(f'<div class="img-label">{title}</div>', unsafe_allow_html=True)
    st.image(image, width="stretch", clamp=True)
    st.caption(caption)


def _require_case(model):
    result = current_analysis(model)
    if result is None:
        note(
            "No radiograph loaded",
            "Upload a PNG or JPEG in the sidebar, or load the synthetic phantom. "
            "The phantom is generated geometry, and any score on it is the model output for that image.",
        )
    return result


def _research_table(rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    formats = {
        column: st.column_config.NumberColumn(format="%.4f")
        for column in frame.columns
        if column not in {"Pipeline", "Tier", "Epoch"}
    }
    if "RPI (%)" in formats:
        formats["RPI (%)"] = st.column_config.NumberColumn(format="%.2f")
    st.dataframe(frame, width="stretch", hide_index=True, column_config=formats)


def _format_audit_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a scannable display DataFrame with formatted 95% CIs and integer confusion counts."""
    out = pd.DataFrame()
    out["Pipeline"] = df["DisplayPipeline"]
    out["T*"] = df["T"].map(lambda v: f"{v:.4f}")
    out["ROC-AUC (95% CI)"] = df.apply(lambda r: f"{r['ROC-AUC']:.4f} ({r['AUC_CI_L']:.4f}–{r['AUC_CI_H']:.4f})", axis=1)
    out["ECE (95% CI)"] = df.apply(lambda r: f"{r['ECE']:.4f} ({r['ECE_CI_L']:.4f}–{r['ECE_CI_H']:.4f})", axis=1)
    out["Brier"] = df["Brier"].map(lambda v: f"{v:.4f}")
    out["NLL"] = df["NLL"].map(lambda v: f"{v:.4f}")
    out["RPI (%)"] = df["RPI (%)"].map(lambda v: f"{v:.2f}")
    out["Sens."] = df["Sensitivity"].map(lambda v: f"{v:.4f}")
    out["Spec."] = df["Specificity"].map(lambda v: f"{v:.4f}")
    out["PPV"] = df["PPV"].map(lambda v: f"{v:.4f}")
    out["NPV"] = df["NPV"].map(lambda v: f"{v:.4f}")
    out["TP/TN/FP/FN"] = df.apply(lambda r: f"{int(r['TP'])}/{int(r['TN'])}/{int(r['FP'])}/{int(r['FN'])}", axis=1)
    return out


# ==============================================================================
# PAGE 1: OVERVIEW
# ==============================================================================
def page_overview() -> None:
    audit = get_audit_bundle()
    page_header(
        "Stanford MURA XR_WRIST · Frozen ResNet-50 · CLAIM 2024 Audited",
        "Smart CAAP v2.0",
        "Quality-guided adaptive radiograph routing and post-hoc confidence calibration with zero backbone retraining. "
        "Updated with full Split B symmetric temperature scaling, patient-cluster bootstrap 95% CIs, DeLong/McNemar tests, "
        "and ResNet-50 Grad-CAM failure analysis.",
    )

    if audit["available"]:
        t12 = audit["tables_1_2"]
        full_sym = t12[t12["Cohort"] == "Table_I_Full_Test_Symmetric_T_All_Baselines"]
        caap = full_sym[full_sym["Pipeline"] == "Smart_CAAP_v2"].iloc[0]
        raw = full_sym[full_sym["Pipeline"] == "Raw_Unfiltered"].iloc[0]
        clahe = full_sym[full_sym["Pipeline"] == "CLAHE_Traditional"].iloc[0]
        manifest = audit["manifest"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Split B Test Cohort", "738 studies", delta="693 patients · 2,091 views")
        c2.metric(
            "CAAP ROC-AUC (95% CI)",
            f"{caap['ROC-AUC']:.4f}",
            delta=f"[{caap['AUC_CI_L']:.4f}, {caap['AUC_CI_H']:.4f}]",
        )
        c3.metric(
            "CAAP Sensitivity",
            f"{caap['Sensitivity']:.4f}",
            delta=f"{caap['Sensitivity'] - raw['Sensitivity']:+.4f} vs raw · {caap['Sensitivity'] - clahe['Sensitivity']:+.4f} vs CLAHE",
        )
        c4.metric(
            "CAAP NPV",
            f"{caap['NPV']:.4f}",
            delta=f"{caap['NPV'] - raw['NPV']:+.4f} vs raw",
        )

        note(
            "CLAIM 2024 Audit Highlights (Split B · Symmetric T* · B = 1,000 Patient-Cluster Bootstrap)",
            f"On the isolated Split B test set (n = 738 studies, SHA-256 {manifest.get('split_b_sha256', '')[:12]}…), "
            f"Smart CAAP v2.0 achieves the highest fracture Sensitivity ({caap['Sensitivity']:.4f}, TP=254/290) and NPV ({caap['NPV']:.4f}) "
            f"of all four pipelines while significantly outperforming Blind Global Stretch in ROC-AUC (+0.0147, DeLong p = 0.0011). "
            f"Under moderate synthetic degradation (Tier 2, seed = 42), Smart CAAP v2.0 boosts ROC-AUC to 0.7430 "
            f"(+0.1078 vs Raw, DeLong p = 4.24e-8; +0.0978 vs CLAHE, p = 9.55e-8). "
            f"Under severe noise (Tier 3), Smart CAAP v2.0 holds ECE at 0.0849 vs 0.3792 (CLAHE) and 0.3958 (Blind Stretch).",
        )

        left, right = st.columns((1.35, 0.85))
        with left:
            metric = st.selectbox(
                "Full-test metric (Split B · Symmetric T* · 95% Bootstrap CI)",
                ["ROC-AUC", "Sensitivity", "NPV", "ECE", "Brier", "NLL", "RPI (%)", "Specificity", "PPV"],
            )
            ci_l, ci_h = CI_MAP.get(metric, (None, None))
            st.plotly_chart(
                metric_bars_with_ci(full_sym, metric, ci_l, ci_h, f"Split B Full Test (n = 738) · {metric}"),
                width="stretch",
            )
        with right:
            telemetry = manifest.get("route_telemetry_full_test", {})
            donut_counts = {
                "Path 1 · Clean bypass": telemetry.get("Path_1_Clean_Bypass", 922),
                "Path 2 · ROI stretch": telemetry.get("Path_2_ROI_Stretch", 675),
                "Path 3 · Bilateral + stretch": telemetry.get("Path_3_Bilateral_Stretch", 494),
            }
            st.plotly_chart(routing_donut(donut_counts, "Smart CAAP v2.0 routing on 2,091 test views"), width="stretch")

        a, b, c = st.columns(3)
        with a:
            note(
                "Symmetric Temperatures (Split B Val)",
                f"Raw T* = {SYMMETRIC_TEMPS['Model input']:.4f} · CLAHE T* = {SYMMETRIC_TEMPS['CLAHE']:.4f} · "
                f"Blind Stretch T* = {SYMMETRIC_TEMPS['Blind global stretch']:.4f} · CAAP T* = {SYMMETRIC_TEMPS['CAAP']:.4f}.",
            )
        with b:
            note(
                "ROI Quality Sensors (Split B Val)",
                f"Central 60% ROI thresholds verified at tau_contrast = {manifest.get('tau_contrast_val', TAU_CONTRAST_CAL):.4f} "
                f"and tau_noise = {manifest.get('tau_noise_val', TAU_NOISE_CAL):.6f}.",
            )
        with c:
            note(
                "Latency Transparency",
                "Core NumPy/OpenCV preprocessing path: ~0.11 ms/image. "
                "Full single-thread router end-to-end (Intel Xeon Platinum 8481C): 0.9772 ms/image.",
            )
    else:
        caap = next(row for row in FULL_TEST if row["Pipeline"] == "CAAP")
        raw = next(row for row in FULL_TEST if row["Pipeline"] == "Raw unfiltered")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test studies", f"{SPLIT_COUNTS['Test studies']}")
        c2.metric("CAAP ROC-AUC", f"{caap['ROC-AUC']:.4f}", delta=f"{caap['ROC-AUC'] - raw['ROC-AUC']:+.4f} vs raw")
        c3.metric("CAAP ECE", f"{caap['ECE']:.4f}", delta=f"{caap['ECE'] - raw['ECE']:+.4f} vs raw", delta_color="inverse")
        c4.metric("CAAP RPI", f"{caap['RPI (%)']:.2f}", delta=f"{caap['RPI (%)'] - raw['RPI (%)']:+.2f} vs raw")


# ==============================================================================
# PAGE 2: LIVE X-RAY ANALYSIS (+ LIVE GRAD-CAM)
# ==============================================================================
def page_analysis() -> None:
    page_header(
        "Live inference & Grad-CAM",
        "X-ray analysis",
        "The upload is resized to 320×320, routed through Smart CAAP v2.0, and scored by the frozen ResNet-50. "
        "Switch between Symmetric Split-B Val T* and Original Notebook T, or enable live ResNet-50 layer4 Grad-CAM in the sidebar.",
    )
    result = _require_case(model)
    if result is None:
        return
    if result["synthetic"]:
        st.warning("This image is a synthetic phantom. The probability is the model output on that phantom, not a clinical result.")

    same_input = np.array_equal(result["source"], result["input"])
    changed_pixels = int(np.count_nonzero(result["difference"]))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROI contrast span", f"{result['c_span']:.4f}", delta=f"threshold {TAU_CONTRAST_CAL:.4f}")
    m2.metric("ROI noise energy", f"{result['hf_energy']:.6f}", delta=f"threshold {TAU_NOISE_CAL:.6f}")
    m3.metric("Pixels changed by CAAP", f"{changed_pixels:,}")
    hardware = "Flagged" if result["hardware_flag"] else "Not flagged"
    m4.metric("Hardware proxy", hardware, delta=f"bright area {result['bright_ratio'] * 100:.2f}%")

    score_col, chart_col = st.columns((0.85, 1.15))
    with score_col:
        _score_panel(result)
        st.caption("Research score only. This application does not diagnose a fracture.")
    with chart_col:
        st.plotly_chart(probability_bars(result["pipelines"]), width="stretch")

    st.markdown("##### Views")
    cards = []
    if not same_input:
        cards.append(("Uploaded original", result["source"], "File before the degradation control."))
    cards.extend(
        [
            ("Model input", result["input"], "Image presented to the router and the network."),
            ("CAAP output", result["caap"], result["route"]),
            ("Central ROI", result["roi_overlay"], "Cyan box is the central 60% sensor window."),
            ("Absolute difference", result["difference_color"], f"{changed_pixels:,} pixels differ from the model input."),
        ]
    )
    if result.get("gradcam_overlay") is not None:
        cards.append(("ResNet-50 Grad-CAM", result["gradcam_overlay"], "Live layer4[-1] spatial attention on CAAP output."))

    columns = st.columns(len(cards))
    for column, (title, image, caption) in zip(columns, cards):
        with column:
            _image_block(title, image, caption)

    st.markdown("##### Before and after")
    fraction = st.slider("Divider", 0.0, 1.0, 0.5, 0.01, label_visibility="collapsed")
    left_cap, right_cap = st.columns(2)
    left_cap.caption("Left of the line · model input")
    right_cap.caption("Right of the line · CAAP output")
    st.image(wipe_compare(result["input"], result["caap"], fraction), width="stretch")

    hist_col, table_col = st.columns((0.9, 1.1))
    with hist_col:
        st.plotly_chart(intensity_histogram(result["input"], result["caap"]), width="stretch")
    with table_col:
        table = pd.DataFrame(
            [
                {
                    "Pipeline": item["name"],
                    "Route / filter": item["note"],
                    "Logit": item["logit"],
                    "Temperature": item["temperature"],
                    "Raw probability": item["raw_probability"],
                    "Reported probability": item["probability"],
                    "Operating point": item["decision"],
                    "SSIM": item["ssim"],
                    "PSNR": item["psnr"] if item["psnr"] is not None else np.nan,
                    "ROI span after": item["roi_span"],
                    "ROI noise after": item["roi_hf"],
                }
                for item in result["pipelines"]
            ]
        )
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config={
                "Logit": st.column_config.NumberColumn(format="%.4f"),
                "Temperature": st.column_config.NumberColumn(format="%.4f"),
                "Raw probability": st.column_config.NumberColumn(format="%.4f"),
                "Reported probability": st.column_config.NumberColumn(format="%.4f"),
                "SSIM": st.column_config.NumberColumn(format="%.4f"),
                "PSNR": st.column_config.NumberColumn(format="%.2f"),
                "ROI span after": st.column_config.NumberColumn(format="%.4f"),
                "ROI noise after": st.column_config.NumberColumn(format="%.6f"),
            },
        )
        png_ok, png_bytes = cv2.imencode(".png", result["caap"])
        if png_ok:
            st.download_button(
                "Download CAAP PNG",
                data=png_bytes.tobytes(),
                file_name=f"caap_{result['image_name']}",
                mime="image/png",
            )


# ==============================================================================
# PAGE 3: CAAP PROCESSING
# ==============================================================================
def _path_card(title: str, rule: str, active: bool, detail: str) -> None:
    state = "Selected" if active else "Not selected"
    klass = "path-card active" if active else "path-card"
    st.markdown(
        f"""
        <div class="{klass}">
          <div class="state">{state}</div>
          <h3>{title}</h3>
          <p>{rule}</p>
          <p>{detail}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_processing() -> None:
    page_header(
        "Signal path",
        "CAAP processing",
        "The router reads two numbers from the central 60% skeletal window, then applies one of three operators. "
        "The branch you see is the branch the code took for the loaded image.",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        _path_card(
            "Path 1 · clean bypass",
            f"Span ≥ {TAU_CONTRAST_CAL:.4f} and noise ≤ {TAU_NOISE_CAL:.6f}",
            False,
            "The pixels are copied through. This is the identity path.",
        )
    with c2:
        _path_card(
            "Path 2 · ROI stretch",
            f"Noise ≤ {TAU_NOISE_CAL:.6f} and span below the contrast threshold",
            False,
            "P2 and P98 come from the central 60% ROI. Blended at alpha = 0.75.",
        )
    with c3:
        _path_card(
            "Path 3 · denoise then stretch",
            f"Noise above {TAU_NOISE_CAL:.6f}",
            False,
            "Bilateral filter (d=5, sigma=25), then ROI stretch blended at alpha = 0.80.",
        )

    result = _require_case(model)
    if result is None:
        st.latex(r"C_{span}=P_{98}-P_{2}")
        st.latex(r"E_{HF}=\mathrm{Var}(\nabla^{2} I_{ROI})")
        return

    st.markdown("##### Decision for the loaded image")
    d1, d2, d3 = st.columns(3)
    with d1:
        _path_card("Path 1 · clean bypass", "Identity", result["route"] == PATH_BYPASS, f"Measured span {result['c_span']:.4f}")
    with d2:
        _path_card("Path 2 · ROI stretch", "Soft blend 0.75", result["route"] == PATH_STRETCH, f"Measured noise {result['hf_energy']:.6f}")
    with d3:
        _path_card("Path 3 · denoise then stretch", "Bilateral, then blend 0.80", result["route"] == PATH_DENOISE, result["route"])

    st.markdown("##### Intermediates")
    stages = [
        ("1 · Model input", result["input"], "Grayscale, 320×320."),
        ("2 · ROI crop", result["roi"], "Central 60% sensor window. Lead markers and borders stay outside."),
    ]
    if result["denoised"] is not None:
        stages.append(("3 · Bilateral filter", result["denoised"], "d = 5, sigma color = 25, sigma space = 25."))
    stages.append((f"{len(stages) + 1} · Routed output", result["caap"], result["route"]))
    columns = st.columns(len(stages))
    for column, (title, image, caption) in zip(columns, stages):
        with column:
            _image_block(title, image, caption)

    st.latex(r"I_{out}=\mathrm{round}\!\left(\alpha I_{stretch}+(1-\alpha) I\right)")
    compare = st.columns(3)
    with compare[0]:
        _image_block("Comparison · CLAHE", result["clahe"], "Fixed 8×8 tiles, clip 2.0.")
    with compare[1]:
        _image_block("Comparison · global stretch", result["stretch"], "Global P1–P99 (sensitive to bright lead markers).")
    with compare[2]:
        _image_block("Routed Smart CAAP v2.0", result["caap"], "Tensor scored by the frozen ResNet-50.")


# ==============================================================================
# PAGE 4: MODEL PERFORMANCE (SPLIT B TABLES I & II + 95% CIs + CONFUSION MATRIX)
# ==============================================================================
def page_performance() -> None:
    audit = get_audit_bundle()
    page_header(
        "Split B evaluation · B = 1,000 patient-cluster bootstrap",
        "Model performance (Tables I & II)",
        "Audited metrics on the isolated Split B test set (n = 738 studies) and Natural Degraded cohort (n = 518 studies) "
        "with symmetric validation temperature scaling, 95% confidence intervals, Brier/NLL, PPV/NPV, and exact confusion matrices.",
    )

    if audit["available"]:
        t12 = audit["tables_1_2"]
        cal_choice = st.radio(
            "Calibration regime",
            options=["Symmetric T* (All Baselines Calibrated)", "Uncalibrated Baselines (T = 1.0)"],
            horizontal=True,
        )
        suffix = "Symmetric_T_All_Baselines" if "Symmetric" in cal_choice else "Uncalibrated_Baselines"

        df_t1 = t12[t12["Cohort"] == f"Table_I_Full_Test_{suffix}"].reset_index(drop=True)
        df_t2 = t12[t12["Cohort"] == f"Table_II_Natural_Degraded_{suffix}"].reset_index(drop=True)

        st.markdown("##### Table I · Full Isolated Split B Test Set (n = 738 studies, 693 patients)")
        m_col, cm_col = st.columns((1.35, 0.85))
        with m_col:
            metric = st.selectbox(
                "Metric with 95% bootstrap CI",
                ["ROC-AUC", "ECE", "Brier", "NLL", "RPI (%)", "Sensitivity", "Specificity", "PPV", "NPV"],
                key="perf_metric_t1",
            )
            ci_l, ci_h = CI_MAP.get(metric, (None, None))
            st.plotly_chart(
                metric_bars_with_ci(df_t1, metric, ci_l, ci_h, f"Table I (n = 738) · {metric}"),
                width="stretch",
            )
        with cm_col:
            pipe_pick = st.selectbox(
                "Inspect confusion matrix (threshold = 0.50)",
                df_t1["DisplayPipeline"].tolist(),
                index=3,
                key="cm_pipe_t1",
            )
            row = df_t1[df_t1["DisplayPipeline"] == pipe_pick].iloc[0]
            st.plotly_chart(
                confusion_matrix_heatmap(
                    int(row["TP"]), int(row["TN"]), int(row["FP"]), int(row["FN"]),
                    f"{pipe_pick} · Confusion Matrix (n = 738)",
                ),
                width="stretch",
            )
            st.caption(
                f"Sens = {row['Sensitivity']:.4f} · Spec = {row['Specificity']:.4f} · "
                f"PPV = {row['PPV']:.4f} · NPV = {row['NPV']:.4f} · F1 = {row['F1-Score']:.4f}"
            )
        st.dataframe(_format_audit_table(df_t1), width="stretch", hide_index=True)

        st.markdown("##### Table II · Natural Degraded Cohort (n = 518 studies)")
        st.dataframe(_format_audit_table(df_t2), width="stretch", hide_index=True)

        left, right = st.columns(2)
        with left:
            st.plotly_chart(training_figure(TRAINING_LOG), width="stretch")
        with right:
            st.plotly_chart(routing_donut(ROUTING_FULL_TEST, "Full test · 2,091 views routed"), width="stretch")

        with st.expander("View original notebook printout tables (pre-audit archive)"):
            note("Original notebook run (T = 1.2215 on CAAP only)", FULL_TEST_NOTE)
            _research_table(FULL_TEST)
            note("Original natural cohort printout", NATURAL_COHORT_NOTE)
            _research_table(NATURAL_COHORT)
    else:
        note("Full test", FULL_TEST_NOTE)
        _research_table(FULL_TEST)


# ==============================================================================
# PAGE 5: EXPERIMENTAL COMPARISON (SEEDED TABLE V STRESS TEST)
# ==============================================================================
def page_comparison() -> None:
    audit = get_audit_bundle()
    page_header(
        "Deterministic seeded stress test (seed = 42) & earlier router",
        "Experimental comparison (Table V)",
        "Deterministic power-law fading + Gaussian noise stress test across all 738 Split B test studies (Tiers 0–3), "
        "with 95% patient-cluster bootstrap CIs and symmetric temperature scaling.",
    )

    if audit["available"]:
        t5 = audit["table_5"]
        mode = st.radio(
            "Stress-test calibration mode",
            options=["Symmetric_T", "Asymmetric_T"],
            format_func=lambda x: "Symmetric T* (All Baselines)" if x == "Symmetric_T" else "Asymmetric T (CAAP Only)",
            horizontal=True,
        )
        df_mode = t5[t5["Calibration_Mode"] == mode].reset_index(drop=True)
        metric = st.selectbox(
            "Stress-test metric (with 95% bootstrap CI)",
            ["ROC-AUC", "ECE", "Brier", "NLL", "RPI (%)", "Sensitivity", "Specificity", "PPV", "NPV"],
            key="tier_metric_audit",
        )
        ci_l, ci_h = CI_MAP.get(metric, (None, None))
        st.plotly_chart(
            tier_lines(df_mode, metric, ci_l, ci_h, f"Table V Seeded Stress Test (seed=42, {mode}) · {metric}"),
            width="stretch",
        )

        note(
            "Key Stress-Test Findings (Seed = 42 · Verified by DeLong & Bootstrap)",
            "Tier 2 (Moderate Fading + Noise): Smart CAAP v2.0 achieves the highest ROC-AUC at 0.7430 "
            "(95% CI 0.7068–0.7813), outperforming Raw (+0.1078, DeLong p = 4.24e-8), CLAHE (+0.0978, p = 9.55e-8), "
            "and Blind Stretch (+0.1646, p = 7.18e-10), with 91.72% Sensitivity and 81.95% NPV. "
            "Tier 3 (Severe Fading + Heavy Noise): CLAHE and Blind Stretch suffer catastrophic calibration collapse "
            "(ECE = 0.3792 and 0.3958, Specificity = 0.0000 and 0.0156), whereas Smart CAAP v2.0 preserves calibration "
            "(ECE = 0.0849, bootstrap ΔECE p < 0.001) and Specificity (0.7321).",
        )

        tier_filter = st.selectbox("Inspect tier table", sorted(df_mode["Tier"].unique().tolist()))
        st.dataframe(_format_audit_table(df_mode[df_mode["Tier"] == tier_filter]), width="stretch", hide_index=True)
    else:
        note("Synthetic protocol", SYNTHETIC_NOTE)
        st.plotly_chart(tier_lines(pd.DataFrame(SYNTHETIC_TIERS), "ROC-AUC"), width="stretch")
        _research_table(SYNTHETIC_TIERS)

    st.markdown("##### Earlier router (v1.0 archive)")
    note("Separate experiment", EARLIER_ROUTER_NOTE)
    st.plotly_chart(metric_bars(EARLIER_ROUTER, "ROC-AUC", "Earlier router · ROC-AUC"), width="stretch")
    _research_table(EARLIER_ROUTER)


# ==============================================================================
# PAGE 6: CLAIM 2024 AUDIT (PARTITIONS + DELONG / MCNEMAR / BOOTSTRAP + LATEX)
# ==============================================================================
def page_claim_audit() -> None:
    audit = get_audit_bundle()
    page_header(
        "IEEE / CLAIM 2024 Reproducibility Verification",
        "CLAIM 2024 statistical & partition audit",
        "Exact patient-disjoint dataset governance (CLAIM Items 35–36), pairwise correlated DeLong ROC-AUC tests, "
        "McNemar's paired prediction tests, paired patient-cluster bootstrap deltas, and copy-pasteable IEEEtran LaTeX exports.",
    )
    if not audit["available"]:
        st.info("Run `python -m experiments.run_claim_audit_extensions --mura-root <MURA_PATH>` to populate audit results.")
        return

    manifest = audit["manifest"]
    st.markdown("##### 1. Dataset & Partition Summary (CLAIM Items 35–36)")
    h1, h2 = st.columns(2)
    h1.code(f"Split A SHA-256: {manifest.get('split_a_sha256', 'N/A')}", language="text")
    h2.code(f"Split B SHA-256: {manifest.get('split_b_sha256', 'N/A')}", language="text")
    st.dataframe(audit["partitions"], width="stretch", hide_index=True)

    st.markdown("##### 2. Pairwise Statistical Significance (DeLong · McNemar · Paired Bootstrap)")
    sig_df = audit["significance"]
    cohorts = sig_df["Cohort"].unique().tolist()
    selected_cohort = st.selectbox("Select audited cohort", cohorts)
    sub_sig = sig_df[sig_df["Cohort"] == selected_cohort].copy()
    st.dataframe(
        sub_sig,
        width="stretch",
        hide_index=True,
        column_config={
            "Delta_AUC": st.column_config.NumberColumn(format="%+.4f"),
            "DeLong_z": st.column_config.NumberColumn(format="%.3f"),
            "DeLong_p": st.column_config.NumberColumn(format="%.4e"),
            "Boot_Delta_ECE_Reduction": st.column_config.NumberColumn(format="%+.4f"),
            "Boot_Delta_ECE_p": st.column_config.NumberColumn(format="%.4f"),
            "Boot_Delta_Brier_Reduction": st.column_config.NumberColumn(format="%+.4f"),
            "Boot_Delta_NLL_Reduction": st.column_config.NumberColumn(format="%+.4f"),
            "Boot_Delta_RPI": st.column_config.NumberColumn(format="%+.2f"),
            "McNemar_p": st.column_config.NumberColumn(format="%.4e"),
        },
    )

    st.markdown("##### 3. Export Publication-Ready IEEE LaTeX & Audit Manifest")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Download IEEE Tables (.tex)",
            data=audit["latex_tables"],
            file_name="ieee_latex_tables.tex",
            mime="text/plain",
            width="stretch",
        )
    with d2:
        st.download_button(
            "Download Partition Table (.tex)",
            data=audit["latex_partitions"],
            file_name="dataset_partition_summary.tex",
            mime="text/plain",
            width="stretch",
        )
    with d3:
        st.download_button(
            "Download Audit Manifest (.json)",
            data=audit["manifest_raw"],
            file_name="audit_manifest.json",
            mime="application/json",
            width="stretch",
        )


# ==============================================================================
# PAGE 7: GRAD-CAM FAILURE ANALYSIS (CLAIM ITEMS 31 & 39)
# ==============================================================================
def page_gradcam_failures() -> None:
    audit = get_audit_bundle()
    page_header(
        "Explainability & Error Audit · CLAIM 2024 Items 31 & 39",
        "Grad-CAM failure analysis",
        "High-resolution 3-panel comparisons (Original + 60% ROI, Smart CAAP v2.0 output, and frozen ResNet-50 layer4 Grad-CAM) "
        "for the 5 highest-confidence False Negatives and 5 highest-confidence False Positives on the Split B MURA wrist test set.",
    )
    if not audit["available"] or audit["gradcam"].empty:
        st.info("No bundled Grad-CAM failure cases found. Run `experiments/gradcam_failure_analysis.py` to generate them.")
        return

    df_gc = audit["gradcam"]
    n_fn = int((df_gc["error_type"] == "False Negative").sum())
    n_fp = int((df_gc["error_type"] == "False Positive").sum())
    fp_hw = int(((df_gc["error_type"] == "False Positive") & (df_gc["hardware_flag"] == True)).sum())
    fn_hw = int(((df_gc["error_type"] == "False Negative") & (df_gc["hardware_flag"] == True)).sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Audited False Negatives", f"{n_fn}")
    k2.metric("Audited False Positives", f"{n_fp}")
    k3.metric("FPs with Metallic Hardware", f"{fp_hw}/{n_fp}", delta="100% hardware-associated")
    k4.metric("FNs with Metallic Hardware", f"{fn_hw}/{n_fn}", delta="All 5 took Path 1 bypass")

    note(
        "Clinical Failure-Mode Finding (Shortcut Learning vs. Subtle Non-Displaced Fractures)",
        "Every top False Positive (5/5) was flagged by the Sobel metallic-implant detector (hardware_flag = True), "
        "confirming that orthopedic plates/screws act as a strong positive shortcut cue for the frozen ResNet-50 backbone. "
        "Conversely, all 5 top False Negatives had no hardware and passed cleanly through Path 1 (Clean Bypass), "
        "indicating subtle or occult non-displaced fractures where preprocessing did not distort the image.",
    )

    f1, f2 = st.columns(2)
    with f1:
        err_filter = st.radio("Filter error type", ["All", "False Negative", "False Positive"], horizontal=True)
    with f2:
        hw_filter = st.radio("Filter hardware flag", ["All", "Hardware Flagged", "No Hardware"], horizontal=True)

    filtered = df_gc.copy()
    if err_filter != "All":
        filtered = filtered[filtered["error_type"] == err_filter]
    if hw_filter == "Hardware Flagged":
        filtered = filtered[filtered["hardware_flag"] == True]
    elif hw_filter == "No Hardware":
        filtered = filtered[filtered["hardware_flag"] == False]

    for _, row in filtered.iterrows():
        img_path = Path(str(row["local_image_path"]))
        st.markdown(
            f"**{row['error_type']} · `{row['study_id']}`** — "
            f"Study P(fx) = `{row['study_prob']:.4f}` · View P(fx) = `{row['view_prob']:.4f}` · "
            f"Route: `{row['route']}` · Hardware: `{'Flagged' if row['hardware_flag'] else 'None'}`"
        )
        if img_path.is_file():
            st.image(str(img_path), width="stretch")
        else:
            st.warning(f"Image file missing: {img_path.name}")

    st.markdown("##### Failure Case Metadata")
    st.dataframe(
        df_gc.drop(columns=["figure_file", "local_image_path"], errors="ignore"),
        width="stretch",
        hide_index=True,
    )


# ==============================================================================
# PAGE 8: RESEARCH METHODOLOGY
# ==============================================================================
def page_methodology() -> None:
    page_header(
        "What was held fixed",
        "Research methodology",
        "The workstation calls the same ROI-anchored router, frozen ResNet-50 checkpoint, and CLAIM 2024 evaluation protocols "
        "implemented in caap/ and experiments/.",
    )
    st.markdown("##### Data and patient-disjoint splits (Split A & Split B)")
    st.markdown(
        "Evaluated on Stanford MURA v1.1 `XR_WRIST` (3,474 unique patients, 3,697 studies, 10,411 views). "
        "Split A uses patient-stratified `train_test_split` (seed=42; 740 test studies). "
        "Split B uses `StratifiedGroupKFold` (5-fold then 8-fold, seed=42; 2,588 train, 371 validation, 738 test studies) "
        "with zero patient overlap verified programmatically and locked by SHA-256 hashes."
    )
    st.markdown("##### Frozen ResNet-50 backbone & study pooling")
    st.latex(r"z_{\text{study}}=\frac{1}{V}\sum_{v=1}^{V} z_v")
    st.markdown(
        f"ResNet-50 (`best_resnet50_wrist.pth`, ImageNet-initialized, 1-logit head, AdamW lr=1e-4, cosine schedule, "
        f"best clean val ROC-AUC = {BEST_VAL_AUC:.4f}) remains strictly frozen across all routing and calibration stages."
    )
    st.markdown("##### Central 60% ROI quality sensors & tri-path router")
    st.latex(r"C_{\text{span}}=P_{98}(I_{\text{ROI}})-P_{2}(I_{\text{ROI}}), \qquad E_{\text{HF}}=\mathrm{Var}\!\left(\nabla^{2} I_{\text{ROI}}\right)")
    st.markdown(
        f"Thresholds calibrated strictly on Split B Validation: `tau_contrast = {TAU_CONTRAST_CAL:.4f}` (35th percentile) "
        f"and `tau_noise = {TAU_NOISE_CAL:.6f}` (75th percentile). "
        "Path 1 bypasses clean views; Path 2 applies ROI-anchored P2–P98 stretching blended at α=0.75; "
        "Path 3 applies edge-preserving bilateral denoising (d=5, σ_color=25, σ_space=25) followed by ROI stretch at α=0.80."
    )
    st.markdown("##### Symmetric post-hoc temperature scaling & statistical inference")
    st.latex(r"p = \sigma\!\left(z_{\text{study}} / T^*\right), \qquad T^* = \arg\min_{T \in [0.2, 3.0]} \mathrm{NLL}_{\text{val}}(T)")
    st.markdown(
        "To eliminate asymmetric calibration bias, `experiments/run_claim_audit_extensions.py` fits `T*` via `L-BFGS-B` "
        "independently on Split B Validation for all four pipelines (`Raw`, `CLAHE`, `Blind Stretch`, `Smart CAAP v2.0`). "
        "Uncertainty is quantified via patient-cluster bootstrapping (`B = 1,000`, `seed = 42`), correlated exact DeLong tests "
        "(Sun & Xu, 2014), and McNemar's paired prediction tests."
    )


# ==============================================================================
# PAGE 9: ABOUT THE PROJECT
# ==============================================================================
def page_about() -> None:
    page_header(
        "Repository & governance record",
        "About Smart CAAP v2.0",
        "Publication-ready research repository targeting IEEE / CLAIM 2024 reproducibility standards for musculoskeletal fracture AI.",
    )
    left, right = st.columns(2)
    with left:
        note(
            "Modular Architecture",
            "`caap/` provides the production router, frozen ResNet-50 loader, live Grad-CAM, and chart components. "
            "`experiments/` provides standalone one-click CLAIM 2024 audit scripts (`data_splits.py`, `metrics_stats.py`, "
            "`run_claim_audit_extensions.py`, `gradcam_failure_analysis.py`).",
        )
        note(
            "Bundled Audit Snapshot",
            "`assets/claim_audit/` bundles the verified Kaggle Split B outputs (CSVs, JSON manifest, IEEE LaTeX tables, "
            "and 10 300-DPI Grad-CAM FN/FP figures) so the dashboard renders real audited data out of the box.",
        )
    with right:
        note(
            "Clinical Scope & Limits",
            "Trained and validated exclusively on MURA v1.1 wrist radiographs. This workstation is for research and "
            "reproducibility auditing only — it does not diagnose or triage patients.",
        )
        note(
            "Latency Benchmarks",
            "Core preprocessing path: ~0.11 ms/image (NumPy/OpenCV). "
            "Full single-thread router end-to-end (Intel Xeon Platinum 8481C): 0.9772 ms/image.",
        )

    st.markdown("##### Verified Constants")
    constants = pd.DataFrame(
        [
            {"Constant": "Input size", "Value": "320 × 320", "Role": "Training and inference resolution"},
            {"Constant": "Contrast threshold (tau_c)", "Value": f"{TAU_CONTRAST_CAL:.4f}", "Role": "Split B Val 35th percentile of central 60% ROI span"},
            {"Constant": "Noise threshold (tau_n)", "Value": f"{TAU_NOISE_CAL:.6f}", "Role": "Split B Val 75th percentile of ROI Laplacian variance"},
            {"Constant": "Symmetric T* (CAAP / Raw / CLAHE / Stretch)", "Value": "0.9460 / 0.9658 / 1.0558 / 1.2405", "Role": "Split B Val L-BFGS-B NLL temperatures"},
            {"Constant": "Notebook T (Original)", "Value": f"{OPTIMAL_T_CAL:.4f}", "Role": "Original asymmetric notebook temperature"},
            {"Constant": "Operating points", "Value": f"{THRESHOLD_STANDARD:.2f} (Standard) / {YOUDEN_T3:.3f} (Youden)", "Role": "Decision thresholds"},
            {"Constant": "Bootstrap resamples", "Value": "B = 1,000 (seed = 42)", "Role": "Patient-cluster 95% confidence intervals"},
        ]
    )
    st.dataframe(constants, width="stretch", hide_index=True)


# ==============================================================================
# APP BOOTSTRAP & NAVIGATION
# ==============================================================================
try:
    model, model_meta = get_model()
    model_error = None
except Exception as exc:
    model, model_meta = None, None
    model_error = f"Checkpoint could not be loaded: {exc}"

_sidebar(model_meta, model_error)

navigation = st.navigation(
    [
        st.Page(page_overview, title="Overview", default=True),
        st.Page(page_analysis, title="X-ray Analysis"),
        st.Page(page_processing, title="CAAP Processing"),
        st.Page(page_performance, title="Model Performance"),
        st.Page(page_comparison, title="Experimental Comparison"),
        st.Page(page_claim_audit, title="CLAIM 2024 Audit"),
        st.Page(page_gradcam_failures, title="Grad-CAM Failures"),
        st.Page(page_methodology, title="Research Methodology"),
        st.Page(page_about, title="About the Project"),
    ]
)
navigation.run()
