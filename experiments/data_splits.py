"""Exact MURA v1.1 XR_WRIST patient-level split builders (Split A & Split B)
and CLAIM 2024 Items 35-36 dataset partition summary generator.

Split A reproduces Notebook Cell 0 (`train_test_split`, seed=42, 70/10/20).
Split B reproduces Notebook Cell 12 (`StratifiedGroupKFold`, n_splits=5 then 8,
seed=42; Train=2,588, Val=371, Test=738 studies used for Smart CAAP v2.0).
"""

from __future__ import annotations

import glob
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from caap.pipeline import IMG_SIZE, detect_metallic_implant

SPLIT_SEED = 42


def load_study_images(img_paths: list[str], size: int = IMG_SIZE) -> list[np.ndarray]:
    """Load grayscale radiographs and area-resize to the model resolution (320x320)."""
    images: list[np.ndarray] = []
    for path in img_paths:
        raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        resized = cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)
        images.append(resized)
    return images


def build_split_a(mura_root: str | Path, seed: int = SPLIT_SEED) -> dict[str, pd.DataFrame | str]:
    """Reproduce Split A (Notebook Cell 0): patient-stratified train_test_split."""
    mura_root = str(mura_root)
    wrist_dirs: list[str] = []
    if os.path.exists(mura_root):
        for root, dirs, _ in os.walk(mura_root):
            if "XR_WRIST" in dirs:
                wrist_dirs.append(os.path.join(root, "XR_WRIST"))
    wrist_dirs = sorted(set(wrist_dirs))

    records: list[dict] = []
    for wrist_path in wrist_dirs:
        subset_name = os.path.basename(os.path.dirname(wrist_path))
        if not os.path.exists(wrist_path):
            continue
        for patient_folder in sorted(os.listdir(wrist_path)):
            patient_path = os.path.join(wrist_path, patient_folder)
            if not os.path.isdir(patient_path):
                continue
            for study_folder in sorted(os.listdir(patient_path)):
                study_path = os.path.join(patient_path, study_folder)
                if not os.path.isdir(study_path):
                    continue
                label = 1 if "positive" in study_folder.lower() else 0
                images = sorted(glob.glob(os.path.join(study_path, "*.png")))
                if images:
                    records.append(
                        {
                            "orig_subset": subset_name,
                            "patient_id": patient_folder,
                            "study_id": f"{patient_folder}_{study_folder}",
                            "study_path": study_path,
                            "label": label,
                            "img_paths": images,
                            "num_images": len(images),
                        }
                    )

    df_all = pd.DataFrame(records)
    if df_all.empty:
        raise FileNotFoundError(f"No XR_WRIST studies found under {mura_root}.")

    patient_labels = df_all.groupby("patient_id")["label"].max().reset_index()
    train_patients, temp_patients = train_test_split(
        patient_labels["patient_id"],
        test_size=0.30,
        stratify=patient_labels["label"],
        random_state=seed,
    )
    temp_df = patient_labels[patient_labels["patient_id"].isin(temp_patients)]
    val_patients, _ = train_test_split(
        temp_df["patient_id"],
        test_size=(2.0 / 3.0),
        stratify=temp_df["label"],
        random_state=seed,
    )

    train_set, val_set = set(train_patients), set(val_patients)
    df_all["split"] = df_all["patient_id"].apply(
        lambda pid: "train" if pid in train_set else ("val" if pid in val_set else "test")
    )

    canonical_csv = df_all[["orig_subset", "patient_id", "study_id", "label", "num_images", "split"]].to_csv(
        index=False
    )
    sha256_hash = hashlib.sha256(canonical_csv.encode("utf-8")).hexdigest()

    return {
        "train": df_all[df_all["split"] == "train"].reset_index(drop=True),
        "val": df_all[df_all["split"] == "val"].reset_index(drop=True),
        "test": df_all[df_all["split"] == "test"].reset_index(drop=True),
        "all": df_all.reset_index(drop=True),
        "sha256": sha256_hash,
    }


def build_split_b(mura_root: str | Path, seed: int = SPLIT_SEED) -> dict[str, pd.DataFrame | str]:
    """Reproduce Split B (Notebook Cell 12): StratifiedGroupKFold (5-fold then 8-fold)."""
    mura_root = str(mura_root)
    study_rows: list[dict] = []
    for root, _, files in os.walk(mura_root):
        if "xr_wrist" in root.lower() and "study" in os.path.basename(root).lower():
            imgs = sorted(
                os.path.join(root, f)
                for f in files
                if f.lower().endswith((".png", ".jpg")) and not f.startswith(".")
            )
            if imgs:
                parts = root.replace("\\", "/").split("/")
                patient_id = next((p for p in parts if p.lower().startswith("patient")), root)
                study_folder = os.path.basename(root)
                label = 1 if "positive" in study_folder.lower() else 0
                study_rows.append(
                    {
                        "study_path": root,
                        "study_id": f"{patient_id}_{study_folder}",
                        "patient_id": patient_id,
                        "label": label,
                        "img_paths": imgs,
                        "num_images": len(imgs),
                    }
                )

    df = (
        pd.DataFrame(study_rows)
        .drop_duplicates(subset=["study_path"])
        .sort_values("study_path")
        .reset_index(drop=True)
    )
    if df.empty:
        raise FileNotFoundError(f"MURA XR_WRIST dataset not found under {mura_root}.")

    sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(sgkf_test.split(df, df["label"], df["patient_id"]))

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    sgkf_val = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf_val.split(train_val_df, train_val_df["label"], train_val_df["patient_id"]))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    df_all = pd.concat([train_df, val_df, test_df], ignore_index=True)

    canonical_csv = df_all[["patient_id", "study_id", "label", "num_images", "split"]].to_csv(index=False)
    sha256_hash = hashlib.sha256(canonical_csv.encode("utf-8")).hexdigest()

    # Verify zero patient leakage across partitions
    p_tr, p_va, p_te = set(train_df["patient_id"]), set(val_df["patient_id"]), set(test_df["patient_id"])
    assert len(p_tr & p_va) == 0 and len(p_tr & p_te) == 0 and len(p_va & p_te) == 0, (
        "Patient leakage detected across Split B partitions!"
    )

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "all": df_all,
        "sha256": sha256_hash,
    }


def _study_has_hardware(img_paths: list[str], hw_cache: dict[str, bool]) -> bool:
    """Return True if any view in the study triggers the metallic-implant proxy."""
    for path in img_paths:
        if path not in hw_cache:
            raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if raw is None:
                hw_cache[path] = False
            else:
                resized = cv2.resize(raw, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                flagged, _, _ = detect_metallic_implant(resized)
                hw_cache[path] = bool(flagged)
        if hw_cache[path]:
            return True
    return False


def summarize_partitions(
    split_a: dict[str, pd.DataFrame | str],
    split_b: dict[str, pd.DataFrame | str],
    include_hardware: bool = True,
) -> pd.DataFrame:
    """Generate CLAIM 2024 Items 35-36 dataset & partition summary table."""
    hw_cache: dict[str, bool] = {}
    rows: list[dict] = []

    for split_name, split_dict in [("Split A (v1.0, train_test_split)", split_a), ("Split B (v2.0, StratifiedGroupKFold)", split_b)]:
        for part_label, key in [("Train", "train"), ("Validation", "val"), ("Test", "test"), ("Total", "all")]:
            df: pd.DataFrame = split_dict[key]  # type: ignore[assignment]
            n_patients = int(df["patient_id"].nunique())
            n_studies = int(len(df))
            views_per_study = df["num_images"].to_numpy(dtype=np.float64)
            n_views = int(views_per_study.sum())
            n_pos = int((df["label"] == 1).sum())
            n_neg = int((df["label"] == 0).sum())
            pos_pct = 100.0 * n_pos / max(n_studies, 1)
            neg_pct = 100.0 * n_neg / max(n_studies, 1)

            if include_hardware:
                hw_flags = [_study_has_hardware(paths, hw_cache) for paths in df["img_paths"]]
                n_hw = int(sum(hw_flags))
                hw_pct = 100.0 * n_hw / max(n_studies, 1)
            else:
                n_hw, hw_pct = -1, float("nan")

            rows.append(
                {
                    "Split": split_name,
                    "Partition": part_label,
                    "Unique Patients": n_patients,
                    "Studies": n_studies,
                    "Views": n_views,
                    "Views/Study (Mean +/- SD)": f"{views_per_study.mean():.2f} +/- {views_per_study.std(ddof=0):.2f}",
                    "Positive n (%)": f"{n_pos} ({pos_pct:.1f}%)",
                    "Negative n (%)": f"{n_neg} ({neg_pct:.1f}%)",
                    "Hardware-Flagged n (%)": f"{n_hw} ({hw_pct:.1f}%)" if include_hardware else "N/A",
                }
            )
    return pd.DataFrame(rows)


def partition_summary_to_latex(df: pd.DataFrame) -> str:
    """Format the CLAIM Items 35-36 partition summary as an IEEEtran LaTeX table."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Stanford MURA v1.1 \texttt{XR\_WRIST} Patient-Disjoint Partition Summary (CLAIM 2024 Items 35--36).}",
        r"\label{tab:dataset_partitions}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Split & Partition & Patients & Studies & Views & Views/Study & Positive $n$ (\%) & Negative $n$ (\%) & Hardware $n$ (\%) \\",
        r"\midrule",
    ]
    prev_split = None
    for _, r in df.iterrows():
        if prev_split is not None and r["Split"] != prev_split:
            lines.append(r"\midrule")
        prev_split = r["Split"]
        split_tex = str(r["Split"]).replace("_", r"\_")
        views_tex = str(r["Views/Study (Mean +/- SD)"]).replace("+/-", r"$\pm$")
        pos_tex = str(r["Positive n (%)"]).replace("%", r"\%")
        neg_tex = str(r["Negative n (%)"]).replace("%", r"\%")
        hw_tex = str(r["Hardware-Flagged n (%)"]).replace("%", r"\%")
        lines.append(
            f"{split_tex} & {r['Partition']} & {r['Unique Patients']} & {r['Studies']} & "
            f"{r['Views']} & {views_tex} & {pos_tex} & {neg_tex} & {hw_tex} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)
