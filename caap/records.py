"""Metrics transcribed from saved notebook outputs.

Values are the rounded figures printed by the research notebook. They are
not recomputed here, because the MURA image files are not in this folder.
"""

from __future__ import annotations

FULL_TEST_NOTE = (
    "Isolated MURA wrist test set, n = 738 studies. "
    "CAAP uses temperature T = 1.2215. The other pipelines use T = 1. "
    "Decision threshold = 0.50. The study score is the sigmoid of the mean view logit. "
    "Source: notebook final full-test benchmark."
)

FULL_TEST = [
    {"Pipeline": "Raw unfiltered", "ROC-AUC": 0.9140, "ECE": 0.0475, "RPI (%)": 87.05, "Sensitivity": 0.7414, "Specificity": 0.9442, "F1-Score": 0.8113, "Accuracy": 0.8645},
    {"Pipeline": "CLAHE", "ROC-AUC": 0.9036, "ECE": 0.0947, "RPI (%)": 81.80, "Sensitivity": 0.6931, "Specificity": 0.9554, "F1-Score": 0.7867, "Accuracy": 0.8523},
    {"Pipeline": "Blind global stretch", "ROC-AUC": 0.8987, "ECE": 0.0698, "RPI (%)": 83.60, "Sensitivity": 0.7207, "Specificity": 0.9375, "F1-Score": 0.7932, "Accuracy": 0.8523},
    {"Pipeline": "CAAP", "ROC-AUC": 0.9084, "ECE": 0.0405, "RPI (%)": 87.16, "Sensitivity": 0.7552, "Specificity": 0.9174, "F1-Score": 0.8022, "Accuracy": 0.8537},
]

NATURAL_COHORT_NOTE = (
    "Test studies in which at least one view fell below the ROI contrast threshold "
    "or above the noise threshold. n = 518 studies. Same router and T = 1.2215. "
    "Threshold = 0.50. Source: notebook calibrated natural-degradation benchmark."
)

NATURAL_COHORT = [
    {"Pipeline": "Raw unfiltered", "ROC-AUC": 0.9103, "ECE": 0.0477, "RPI (%)": 86.69, "Sensitivity": 0.7500, "Specificity": 0.9362, "F1-Score": 0.8168, "Accuracy": 0.8571},
    {"Pipeline": "CLAHE", "ROC-AUC": 0.9008, "ECE": 0.0923, "RPI (%)": 81.76, "Sensitivity": 0.7000, "Specificity": 0.9497, "F1-Score": 0.7918, "Accuracy": 0.8436},
    {"Pipeline": "Blind global stretch", "ROC-AUC": 0.8995, "ECE": 0.0767, "RPI (%)": 83.05, "Sensitivity": 0.7273, "Specificity": 0.9228, "F1-Score": 0.7940, "Accuracy": 0.8398},
    {"Pipeline": "CAAP", "ROC-AUC": 0.9024, "ECE": 0.0438, "RPI (%)": 86.29, "Sensitivity": 0.7682, "Specificity": 0.8960, "F1-Score": 0.8048, "Accuracy": 0.8417},
]

EARLIER_ROUTER_NOTE = (
    "Earlier router, before ROI-anchored blending. Thresholds were "
    "tau contrast = 0.3098, tau noise = 0.005620, and T = 1.8567, with Gaussian "
    "blur plus global stretch. The notebook found 428 of 738 test studies with "
    "natural degradation and printed this table for that run. It is a separate "
    "experiment from the ROI-anchored results above."
)

EARLIER_ROUTER = [
    {"Pipeline": "Raw unfiltered", "ROC-AUC": 0.9171, "ECE": 0.0380, "RPI (%)": 88.23, "Sensitivity": 0.7914, "Specificity": 0.9212, "F1-Score": 0.8362, "Accuracy": 0.8645},
    {"Pipeline": "CLAHE", "ROC-AUC": 0.9167, "ECE": 0.0697, "RPI (%)": 85.28, "Sensitivity": 0.7594, "Specificity": 0.9378, "F1-Score": 0.8256, "Accuracy": 0.8598},
    {"Pipeline": "Blind global stretch", "ROC-AUC": 0.9172, "ECE": 0.0519, "RPI (%)": 86.96, "Sensitivity": 0.7914, "Specificity": 0.8963, "F1-Score": 0.8222, "Accuracy": 0.8505},
    {"Pipeline": "Earlier CAAP", "ROC-AUC": 0.9156, "ECE": 0.0683, "RPI (%)": 85.31, "Sensitivity": 0.8021, "Specificity": 0.8838, "F1-Score": 0.8219, "Accuracy": 0.8481},
]

SYNTHETIC_NOTE = (
    "Power-law fading plus Gaussian noise, applied to all 738 test studies. "
    "This degradation is not the linear alpha/sigma control in X-ray Analysis. "
    "Tier 0 is the original image and matches the full-test table. "
    "Source: notebook synthetic-tier benchmark."
)

SYNTHETIC_TIERS = [
    {"Tier": "Tier 0 · original", "Pipeline": "Raw unfiltered", "ROC-AUC": 0.9140, "ECE": 0.0475, "RPI (%)": 87.05, "Sensitivity": 0.7414, "Accuracy": 0.8645},
    {"Tier": "Tier 0 · original", "Pipeline": "CLAHE", "ROC-AUC": 0.9036, "ECE": 0.0947, "RPI (%)": 81.80, "Sensitivity": 0.6931, "Accuracy": 0.8523},
    {"Tier": "Tier 0 · original", "Pipeline": "Blind global stretch", "ROC-AUC": 0.8987, "ECE": 0.0698, "RPI (%)": 83.60, "Sensitivity": 0.7207, "Accuracy": 0.8523},
    {"Tier": "Tier 0 · original", "Pipeline": "CAAP", "ROC-AUC": 0.9084, "ECE": 0.0405, "RPI (%)": 87.16, "Sensitivity": 0.7552, "Accuracy": 0.8537},
    {"Tier": "Tier 1 · mild fading", "Pipeline": "Raw unfiltered", "ROC-AUC": 0.9142, "ECE": 0.0420, "RPI (%)": 87.58, "Sensitivity": 0.7483, "Accuracy": 0.8604},
    {"Tier": "Tier 1 · mild fading", "Pipeline": "CLAHE", "ROC-AUC": 0.9074, "ECE": 0.0690, "RPI (%)": 84.48, "Sensitivity": 0.7241, "Accuracy": 0.8523},
    {"Tier": "Tier 1 · mild fading", "Pipeline": "Blind global stretch", "ROC-AUC": 0.8939, "ECE": 0.0785, "RPI (%)": 82.38, "Sensitivity": 0.7310, "Accuracy": 0.8564},
    {"Tier": "Tier 1 · mild fading", "Pipeline": "CAAP", "ROC-AUC": 0.8999, "ECE": 0.0405, "RPI (%)": 86.35, "Sensitivity": 0.7517, "Accuracy": 0.8645},
    {"Tier": "Tier 2 · moderate fade + noise", "Pipeline": "Raw unfiltered", "ROC-AUC": 0.6606, "ECE": 0.3550, "RPI (%)": 42.61, "Sensitivity": 0.0103, "Accuracy": 0.6111},
    {"Tier": "Tier 2 · moderate fade + noise", "Pipeline": "CLAHE", "ROC-AUC": 0.6427, "ECE": 0.2495, "RPI (%)": 48.24, "Sensitivity": 0.0621, "Accuracy": 0.6274},
    {"Tier": "Tier 2 · moderate fade + noise", "Pipeline": "Blind global stretch", "ROC-AUC": 0.5777, "ECE": 0.1350, "RPI (%)": 49.97, "Sensitivity": 0.6103, "Accuracy": 0.5569},
    {"Tier": "Tier 2 · moderate fade + noise", "Pipeline": "CAAP", "ROC-AUC": 0.6680, "ECE": 0.0702, "RPI (%)": 62.11, "Sensitivity": 0.4931, "Accuracy": 0.6626},
    {"Tier": "Tier 3 · severe fade + noise", "Pipeline": "Raw unfiltered", "ROC-AUC": 0.6355, "ECE": 0.2678, "RPI (%)": 46.53, "Sensitivity": 0.0034, "Accuracy": 0.6084},
    {"Tier": "Tier 3 · severe fade + noise", "Pipeline": "CLAHE", "ROC-AUC": 0.5658, "ECE": 0.2261, "RPI (%)": 43.78, "Sensitivity": 1.0000, "Accuracy": 0.3930},
    {"Tier": "Tier 3 · severe fade + noise", "Pipeline": "Blind global stretch", "ROC-AUC": 0.4850, "ECE": 0.3245, "RPI (%)": 32.76, "Sensitivity": 1.0000, "Accuracy": 0.3997},
    {"Tier": "Tier 3 · severe fade + noise", "Pipeline": "CAAP", "ROC-AUC": 0.5995, "ECE": 0.0944, "RPI (%)": 54.29, "Sensitivity": 0.0034, "Accuracy": 0.6084},
]

ROUTING_FULL_TEST = {
    "Path 1 · clean bypass": 922,
    "Path 2 · ROI stretch": 675,
    "Path 3 · bilateral + stretch": 494,
}

ROUTING_NATURAL = {
    "Path 1 · clean bypass": 403,
    "Path 2 · ROI stretch": 675,
    "Path 3 · bilateral + stretch": 494,
}

TRAINING_LOG = [
    {"Epoch": 1, "Train BCE loss": 0.4417, "Validation ROC-AUC": 0.9112},
    {"Epoch": 2, "Train BCE loss": 0.3438, "Validation ROC-AUC": 0.9318},
    {"Epoch": 3, "Train BCE loss": 0.2385, "Validation ROC-AUC": 0.9401},
]

SPLIT_COUNTS = {"Train studies": 2588, "Validation studies": 371, "Test studies": 738}
BEST_VAL_AUC = 0.9401
