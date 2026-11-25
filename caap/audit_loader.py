"""Loader for bundled CLAIM 2024 audit outputs (CSVs, JSON manifest, LaTeX, Grad-CAM)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "assets" / "claim_audit"
GRADCAM_DIR = AUDIT_DIR / "gradcam_failures"

PIPELINE_LABELS = {
    "Raw_Unfiltered": "Raw unfiltered",
    "CLAHE_Traditional": "CLAHE",
    "Blind_Stretch_E5": "Blind global stretch",
    "Smart_CAAP_v2": "Smart CAAP v2.0",
}


def audit_available() -> bool:
    return (AUDIT_DIR / "audit_manifest.json").is_file()


def load_manifest() -> dict:
    path = AUDIT_DIR / "audit_manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_partition_summary() -> pd.DataFrame:
    path = AUDIT_DIR / "dataset_partition_summary.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def load_tables_1_2() -> pd.DataFrame:
    path = AUDIT_DIR / "split_b_tables_1_2_metrics_cis.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["DisplayPipeline"] = df["Pipeline"].map(PIPELINE_LABELS).fillna(df["Pipeline"])
    return df


def load_table_5() -> pd.DataFrame:
    path = AUDIT_DIR / "split_b_table_5_seeded_stress_test.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["DisplayPipeline"] = df["Pipeline"].map(PIPELINE_LABELS).fillna(df["Pipeline"])
    return df


def load_significance() -> pd.DataFrame:
    path = AUDIT_DIR / "split_b_significance_delong_mcnemar_boot.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def load_gradcam_cases() -> pd.DataFrame:
    path = GRADCAM_DIR / "gradcam_failure_cases.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Resolve bundled local PNG path from filename (portable across Kaggle/local machines)
    df["local_image_path"] = df["figure_file"].apply(lambda p: str(GRADCAM_DIR / Path(str(p)).name))
    return df


def load_text_file(filename: str) -> str:
    path = AUDIT_DIR / filename
    return path.read_text(encoding="utf-8") if path.is_file() else ""
