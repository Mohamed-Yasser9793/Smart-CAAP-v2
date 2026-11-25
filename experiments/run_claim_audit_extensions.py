"""One-click CLAIM 2024 reproducibility & statistical audit suite for Smart CAAP v2.0.

Closes all peer-review & CLAIM 2024 audit extensions strictly faithful to the
frozen ResNet-50 checkpoint (`best_resnet50_wrist.pth`) and Smart CAAP v2.0 math:
  1. Symmetric Temperature Scaling (T*) via L-BFGS-B on Split B Validation for
     ALL baselines (Raw Unfiltered, CLAHE, Blind Stretch, and Smart CAAP v2.0).
  2. Patient-Cluster Bootstrap 95% CIs (B=1,000, seed=42), exact DeLong and
     McNemar tests, PPV/NPV, and exact integer TP/TN/FP/FN directly on Split B
     across Tables I (Full Test), II (Natural Degraded), and V (Synthetic Tiers).
  3. Dataset & Partition Summary Table (Split A & Split B, CLAIM Items 35-36).
  4. Deterministic Seeded Synthetic Noise Stress Test (fixing Notebook Cell 18).
  5. Optional ResNet-50 Grad-CAM Failure Analysis (--run-gradcam, CLAIM 31 & 39).

Usage:
    python -m experiments.run_claim_audit_extensions \
        --mura-root /kaggle/input \
        --weights weights/best_resnet50_wrist.pth \
        --output-dir outputs/claim_audit
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from caap.inference import DEVICE, WEIGHTS_PATH, _batch_tensor, load_backbone
from caap.pipeline import (
    IMG_SIZE,
    THRESHOLD_STANDARD,
    apply_clahe,
    apply_edge_preserving_denoise_stretch,
    apply_global_blind_stretch,
    apply_roi_anchored_stretch,
    measure_roi_sensors,
)
from experiments.data_splits import (
    SPLIT_SEED,
    build_split_a,
    build_split_b,
    load_study_images,
    partition_summary_to_latex,
    summarize_partitions,
)
from experiments.gradcam_failure_analysis import run_gradcam_failure_analysis
from experiments.metrics_stats import (
    compute_clinical_metrics,
    delong_roc_test,
    fit_temperature_scaling,
    format_ieee_metrics_latex,
    mcnemar_test,
    paired_patient_cluster_bootstrap,
    patient_cluster_bootstrap_ci,
    sigmoid,
)

PIPELINES = ("Raw_Unfiltered", "CLAHE_Traditional", "Blind_Stretch_E5", "Smart_CAAP_v2")

SYNTHETIC_TIER_NAMES = {
    0: "Tier 0: Clean (Original)",
    1: "Tier 1: Mild Fading",
    2: "Tier 2: Moderate Fading + Noise",
    3: "Tier 3: Severe Fading + Heavy Noise",
}


def set_deterministic_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch seeds for deterministic stress testing."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def route_smart_caap_v2(
    img_u8: np.ndarray,
    tau_contrast: float,
    tau_noise: float,
) -> tuple[np.ndarray, str]:
    """Smart CAAP v2.0 tri-path router using validation-calibrated ROI thresholds."""
    c_span, hf_energy = measure_roi_sensors(img_u8)
    if c_span >= tau_contrast and hf_energy <= tau_noise:
        return img_u8, "Path_1_Clean_Bypass"
    if hf_energy <= tau_noise:
        return apply_roi_anchored_stretch(img_u8), "Path_2_ROI_Stretch"
    return apply_edge_preserving_denoise_stretch(img_u8), "Path_3_Bilateral_Stretch"


def degrade_image_synthetic_seeded(
    img_u8: np.ndarray,
    tier: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Deterministic seeded version of Notebook Cell 18 synthetic stress test.

    Exact power-law and Gaussian noise equations preserved verbatim from Cell 18:
      - Tier 0: Original uint8 image
      - Tier 1: x^1.4 * 0.70 + 0.15
      - Tier 2: x^1.8 * 0.50 + 0.25 + N(0, 0.03)
      - Tier 3: x^2.3 * 0.35 + 0.30 + N(0, 0.08)
    """
    if tier == 0:
        return img_u8.copy()
    x = img_u8.astype(np.float32) / 255.0
    if tier == 1:
        x_deg = np.power(x, 1.4) * 0.7 + 0.15
    elif tier == 2:
        x_deg = np.power(x, 1.8) * 0.5 + 0.25
        x_deg = np.clip(x_deg + rng.normal(0.0, 0.03, size=x_deg.shape), 0.0, 1.0)
    else:
        x_deg = np.power(x, 2.3) * 0.35 + 0.30
        x_deg = np.clip(x_deg + rng.normal(0.0, 0.08, size=x_deg.shape), 0.0, 1.0)
    return np.rint(x_deg * 255.0).astype(np.uint8)


def calibrate_thresholds_on_val(val_df: pd.DataFrame) -> tuple[float, float]:
    """Calibrate TAU_CONTRAST (35th pct) and TAU_NOISE (75th pct) on Split B Validation."""
    spans, noises = [], []
    for _, row in val_df.iterrows():
        for im in load_study_images(row["img_paths"], size=IMG_SIZE):
            cs, hf = measure_roi_sensors(im)
            spans.append(cs)
            noises.append(hf)
    tau_contrast = float(np.percentile(spans, 35))
    tau_noise = float(np.percentile(noises, 75))
    return tau_contrast, tau_noise


def collect_study_logits(
    model: torch.nn.Module,
    df: pd.DataFrame,
    tau_contrast: float,
    tau_noise: float,
    synthetic_tier: int | None = None,
    stress_seed: int = 42,
) -> dict:
    """Run all 4 pipelines through frozen ResNet-50 and pool mean view logits per study."""
    logits_map: dict[str, list[float]] = {p: [] for p in PIPELINES}
    labels: list[int] = []
    patient_ids: list[str] = []
    study_ids: list[str] = []
    natural_degraded_mask: list[bool] = []
    route_telemetry = {"Path_1_Clean_Bypass": 0, "Path_2_ROI_Stretch": 0, "Path_3_Bilateral_Stretch": 0}

    with torch.no_grad():
        for study_idx, (_, row) in enumerate(df.iterrows()):
            raw_imgs = load_study_images(row["img_paths"], size=IMG_SIZE)
            if not raw_imgs:
                continue

            needs_rescue = False
            per_pipe_imgs: dict[str, list[np.ndarray]] = {p: [] for p in PIPELINES}

            for view_idx, raw_u8 in enumerate(raw_imgs):
                if synthetic_tier is not None:
                    view_seed = stress_seed * 100000 + synthetic_tier * 10000 + study_idx * 10 + view_idx
                    rng = np.random.default_rng(view_seed)
                    in_u8 = degrade_image_synthetic_seeded(raw_u8, synthetic_tier, rng)
                else:
                    in_u8 = raw_u8

                c_span, hf_energy = measure_roi_sensors(in_u8)
                if c_span < tau_contrast or hf_energy > tau_noise:
                    needs_rescue = True

                caap_u8, r_name = route_smart_caap_v2(in_u8, tau_contrast, tau_noise)
                route_telemetry[r_name] += 1

                per_pipe_imgs["Raw_Unfiltered"].append(in_u8)
                per_pipe_imgs["CLAHE_Traditional"].append(apply_clahe(in_u8))
                per_pipe_imgs["Blind_Stretch_E5"].append(apply_global_blind_stretch(in_u8))
                per_pipe_imgs["Smart_CAAP_v2"].append(caap_u8)

            for p_name in PIPELINES:
                batch_t = _batch_tensor(per_pipe_imgs[p_name])
                view_logits = model(batch_t).view(-1).cpu().numpy().astype(np.float64)
                logits_map[p_name].append(float(np.mean(view_logits)))

            labels.append(int(row["label"]))
            patient_ids.append(str(row["patient_id"]))
            study_ids.append(str(row["study_id"]))
            natural_degraded_mask.append(needs_rescue)

    return {
        "logits": {k: np.asarray(v, dtype=np.float64) for k, v in logits_map.items()},
        "labels": np.asarray(labels, dtype=int),
        "patient_ids": np.asarray(patient_ids),
        "study_ids": np.asarray(study_ids),
        "natural_degraded_mask": np.asarray(natural_degraded_mask, dtype=bool),
        "route_telemetry": route_telemetry,
    }


def evaluate_cohort_with_stats(
    cohort_name: str,
    logits_dict: dict[str, np.ndarray],
    labels: np.ndarray,
    patient_ids: np.ndarray,
    temps: dict[str, float],
    threshold: float = THRESHOLD_STANDARD,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute metrics, 95% patient-cluster bootstrap CIs, DeLong, and McNemar tests."""
    rows: list[dict] = []
    probs_map: dict[str, np.ndarray] = {}

    for p_name in PIPELINES:
        t_val = float(temps[p_name])
        probs = sigmoid(logits_dict[p_name] / t_val)
        probs_map[p_name] = probs

        m = compute_clinical_metrics(labels, probs, threshold=threshold)
        cis = patient_cluster_bootstrap_ci(patient_ids, labels, probs, threshold=threshold, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "Cohort": cohort_name,
                "Pipeline": p_name,
                "T": t_val,
                **m,
                "AUC_CI_L": cis["ROC-AUC"][0],
                "AUC_CI_H": cis["ROC-AUC"][1],
                "ECE_CI_L": cis["ECE"][0],
                "ECE_CI_H": cis["ECE"][1],
                "Brier_CI_L": cis["Brier"][0],
                "Brier_CI_H": cis["Brier"][1],
                "NLL_CI_L": cis["NLL"][0],
                "NLL_CI_H": cis["NLL"][1],
                "RPI_CI_L": cis["RPI (%)"][0],
                "RPI_CI_H": cis["RPI (%)"][1],
                "Sens_CI_L": cis["Sensitivity"][0],
                "Sens_CI_H": cis["Sensitivity"][1],
                "Spec_CI_L": cis["Specificity"][0],
                "Spec_CI_H": cis["Specificity"][1],
                "PPV_CI_L": cis["PPV"][0],
                "PPV_CI_H": cis["PPV"][1],
                "NPV_CI_L": cis["NPV"][0],
                "NPV_CI_H": cis["NPV"][1],
            }
        )

    sig_rows: list[dict] = []
    caap_probs = probs_map["Smart_CAAP_v2"]
    for baseline in ("Raw_Unfiltered", "CLAHE_Traditional", "Blind_Stretch_E5"):
        base_probs = probs_map[baseline]
        dl = delong_roc_test(labels, caap_probs, base_probs)
        mc = mcnemar_test(labels, caap_probs, base_probs, threshold=threshold)
        pb = paired_patient_cluster_bootstrap(
            patient_ids, labels, caap_probs, base_probs, threshold=threshold, n_boot=n_boot, seed=seed
        )
        sig_rows.append(
            {
                "Cohort": cohort_name,
                "Comparison": f"Smart_CAAP_v2 vs {baseline}",
                "Delta_AUC": dl["delta_auc"],
                "DeLong_z": dl["z_stat"],
                "DeLong_p": dl["p_value"],
                "Boot_Delta_AUC_CI": f"[{pb['delta_auc']['ci_low']:+.4f}, {pb['delta_auc']['ci_high']:+.4f}]",
                "Boot_Delta_ECE_Reduction": pb["delta_ece"]["mean_diff"],
                "Boot_Delta_ECE_CI": f"[{pb['delta_ece']['ci_low']:+.4f}, {pb['delta_ece']['ci_high']:+.4f}]",
                "Boot_Delta_ECE_p": pb["delta_ece"]["p_two_sided"],
                "Boot_Delta_Brier_Reduction": pb["delta_brier"]["mean_diff"],
                "Boot_Delta_NLL_Reduction": pb["delta_nll"]["mean_diff"],
                "Boot_Delta_RPI": pb["delta_rpi"]["mean_diff"],
                "Boot_Delta_RPI_CI": f"[{pb['delta_rpi']['ci_low']:+.2f}, {pb['delta_rpi']['ci_high']:+.2f}]",
                "McNemar_b": mc["b_a_right_b_wrong"],
                "McNemar_c": mc["c_a_wrong_b_right"],
                "McNemar_p": mc["p_value"],
                "McNemar_method": mc["method"],
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(sig_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart CAAP v2.0 CLAIM 2024 Reproducibility & Audit Extensions")
    parser.add_argument("--mura-root", type=str, required=True, help="Root path containing Stanford MURA v1.1.")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_PATH), help="Path to best_resnet50_wrist.pth.")
    parser.add_argument("--output-dir", type=str, default="outputs/claim_audit", help="Directory for output artifacts.")
    parser.add_argument("--n-boot", type=int, default=1000, help="Patient-cluster bootstrap resamples (default 1000).")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED, help="Global seed for splits and bootstrap (default 42).")
    parser.add_argument(
        "--stress-seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Deterministic seed(s) for Cell 18 synthetic stress test (e.g. --stress-seeds 42 123 456 789 2024).",
    )
    parser.add_argument("--skip-hardware", action="store_true", help="Skip hardware scan in dataset summary for speed.")
    parser.add_argument("--run-gradcam", action="store_true", help="Also generate Grad-CAM FN/FP failure figures.")
    parser.add_argument("--gradcam-top-k", type=int, default=5, help="Top-K FN and FP studies for Grad-CAM.")
    args = parser.parse_args()

    set_deterministic_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Dataset & Partition Summary (Split A & Split B, CLAIM Items 35-36)
    # ------------------------------------------------------------------
    print("[1/5] Building exact Split A & Split B partitions and CLAIM Items 35-36 summary...")
    split_a = build_split_a(args.mura_root, seed=args.seed)
    split_b = build_split_b(args.mura_root, seed=args.seed)
    part_df = summarize_partitions(split_a, split_b, include_hardware=not args.skip_hardware)
    part_df.to_csv(out_dir / "dataset_partition_summary.csv", index=False)
    (out_dir / "dataset_partition_summary.tex").write_text(partition_summary_to_latex(part_df), encoding="utf-8")
    print(part_df.to_string(index=False))

    val_df: pd.DataFrame = split_b["val"]  # type: ignore[assignment]
    test_df: pd.DataFrame = split_b["test"]  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # 2. Validation Thresholds & Symmetric Temperature Scaling (T*)
    # ------------------------------------------------------------------
    print("\n[2/5] Calibrating ROI thresholds & symmetric L-BFGS-B temperature scaling (T*) on Split B Val...")
    model, meta = load_backbone(Path(args.weights))
    model.to(DEVICE).eval()
    print(f"      Loaded checkpoint: {meta['filename']} ({meta['parameters']:,} params on {meta['device']})")

    tau_c, tau_n = calibrate_thresholds_on_val(val_df)
    print(f"      Split B Val ROI thresholds: tau_contrast={tau_c:.4f}, tau_noise={tau_n:.6f}")

    val_eval = collect_study_logits(model, val_df, tau_c, tau_n)
    symmetric_temps: dict[str, float] = {}
    for p_name in PIPELINES:
        t_star = fit_temperature_scaling(val_eval["logits"][p_name], val_eval["labels"])
        symmetric_temps[p_name] = t_star
        print(f"      Fitted T* ({p_name:<18s}) = {t_star:.4f}")

    original_asymmetric_temps = {
        "Raw_Unfiltered": 1.0,
        "CLAHE_Traditional": 1.0,
        "Blind_Stretch_E5": 1.0,
        "Smart_CAAP_v2": symmetric_temps["Smart_CAAP_v2"],
    }

    # ------------------------------------------------------------------
    # 3. Tables I & II on Split B (Full Test n=738 & Natural Degraded n=518)
    # ------------------------------------------------------------------
    print("\n[3/5] Running Split B Full Test (Table I) & Natural Degraded Cohort (Table II) with CIs/DeLong/McNemar...")
    test_eval = collect_study_logits(model, test_df, tau_c, tau_n)
    nat_mask = test_eval["natural_degraded_mask"]
    print(f"      Full Test n={len(test_eval['labels'])} | Natural Degraded n={int(nat_mask.sum())}")
    print(f"      Routing Telemetry (Full Test): {test_eval['route_telemetry']}")

    all_metric_tables: list[pd.DataFrame] = []
    all_sig_tables: list[pd.DataFrame] = []
    latex_blocks: list[str] = []

    for mode_label, temp_map in [("Uncalibrated_Baselines", original_asymmetric_temps), ("Symmetric_T_All_Baselines", symmetric_temps)]:
        # Table I: Full Test Set
        df_t1, sig_t1 = evaluate_cohort_with_stats(
            f"Table_I_Full_Test_{mode_label}",
            test_eval["logits"],
            test_eval["labels"],
            test_eval["patient_ids"],
            temp_map,
            threshold=THRESHOLD_STANDARD,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        all_metric_tables.append(df_t1)
        all_sig_tables.append(sig_t1)
        latex_blocks.append(
            format_ieee_metrics_latex(
                df_t1,
                caption=f"Split B Full Test Set ($n={len(test_eval['labels'])}$) --- {mode_label.replace('_', ' ')}.",
                label=f"tab:splitb_table1_{mode_label.lower()}",
            )
        )

        # Table II: Natural Degraded Cohort
        nat_logits = {k: v[nat_mask] for k, v in test_eval["logits"].items()}
        df_t2, sig_t2 = evaluate_cohort_with_stats(
            f"Table_II_Natural_Degraded_{mode_label}",
            nat_logits,
            test_eval["labels"][nat_mask],
            test_eval["patient_ids"][nat_mask],
            temp_map,
            threshold=THRESHOLD_STANDARD,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        all_metric_tables.append(df_t2)
        all_sig_tables.append(sig_t2)
        latex_blocks.append(
            format_ieee_metrics_latex(
                df_t2,
                caption=f"Split B Natural Degraded Cohort ($n={int(nat_mask.sum())}$) --- {mode_label.replace('_', ' ')}.",
                label=f"tab:splitb_table2_{mode_label.lower()}",
            )
        )

    # ------------------------------------------------------------------
    # 4. Table V: Deterministic Seeded Synthetic Stress Test (Fixing Cell 18)
    # ------------------------------------------------------------------
    print(f"\n[4/5] Running Deterministic Seeded Synthetic Stress Test (seeds={args.stress_seeds})...")
    stress_rows: list[pd.DataFrame] = []
    for s_seed in args.stress_seeds:
        set_deterministic_seed(s_seed)
        for tier_idx in range(4):
            tier_eval = collect_study_logits(
                model, test_df, tau_c, tau_n, synthetic_tier=tier_idx, stress_seed=s_seed
            )
            for mode_label, temp_map in [("Asymmetric_T", original_asymmetric_temps), ("Symmetric_T", symmetric_temps)]:
                df_tier, sig_tier = evaluate_cohort_with_stats(
                    f"Table_V_Tier{tier_idx}_seed{s_seed}_{mode_label}",
                    tier_eval["logits"],
                    tier_eval["labels"],
                    tier_eval["patient_ids"],
                    temp_map,
                    threshold=THRESHOLD_STANDARD,
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
                df_tier["Tier"] = SYNTHETIC_TIER_NAMES[tier_idx]
                df_tier["Stress_Seed"] = s_seed
                df_tier["Calibration_Mode"] = mode_label
                stress_rows.append(df_tier)
                all_sig_tables.append(sig_tier)
                if s_seed == args.stress_seeds[0] and mode_label == "Symmetric_T":
                    latex_blocks.append(
                        format_ieee_metrics_latex(
                            df_tier,
                            caption=f"Table V ({SYNTHETIC_TIER_NAMES[tier_idx]}, seed={s_seed}, Symmetric $T^*$).",
                            label=f"tab:splitb_table5_tier{tier_idx}_sym",
                        )
                    )

    df_tables_1_2 = pd.concat(all_metric_tables, ignore_index=True)
    df_table_5 = pd.concat(stress_rows, ignore_index=True)
    df_sig_all = pd.concat(all_sig_tables, ignore_index=True)

    df_tables_1_2.to_csv(out_dir / "split_b_tables_1_2_metrics_cis.csv", index=False)
    df_table_5.to_csv(out_dir / "split_b_table_5_seeded_stress_test.csv", index=False)
    df_sig_all.to_csv(out_dir / "split_b_significance_delong_mcnemar_boot.csv", index=False)
    (out_dir / "ieee_latex_tables.tex").write_text("\n\n".join(latex_blocks), encoding="utf-8")

    # Multi-seed summary if >1 stress seed was requested
    if len(args.stress_seeds) > 1:
        agg = (
            df_table_5.groupby(["Calibration_Mode", "Tier", "Pipeline"])[
                ["ROC-AUC", "ECE", "Brier", "NLL", "RPI (%)", "Sensitivity", "Specificity", "PPV", "NPV"]
            ]
            .agg(["mean", "std"])
            .reset_index()
        )
        agg.to_csv(out_dir / "split_b_table_5_multiseed_mean_sd.csv", index=False)

    manifest = {
        "split_a_sha256": split_a["sha256"],
        "split_b_sha256": split_b["sha256"],
        "tau_contrast_val": tau_c,
        "tau_noise_val": tau_n,
        "symmetric_temperatures_val": symmetric_temps,
        "route_telemetry_full_test": test_eval["route_telemetry"],
        "n_boot": args.n_boot,
        "seed": args.seed,
        "stress_seeds": args.stress_seeds,
    }
    (out_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # 5. Optional Grad-CAM Failure Analysis (CLAIM Items 31 & 39)
    # ------------------------------------------------------------------
    if args.run_gradcam:
        print("\n[5/5] Generating ResNet-50 Grad-CAM Failure Analysis figures (CLAIM Items 31 & 39)...")
        run_gradcam_failure_analysis(
            mura_root=args.mura_root,
            weights_path=args.weights,
            output_dir=out_dir / "gradcam_failures",
            top_k=args.gradcam_top_k,
            threshold=THRESHOLD_STANDARD,
            temperature=symmetric_temps["Smart_CAAP_v2"],
        )
    else:
        print("\n[5/5] Skipping Grad-CAM (pass --run-gradcam to generate FN/FP heatmaps).")

    print(f"\n[DONE] All CLAIM 2024 audit outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
