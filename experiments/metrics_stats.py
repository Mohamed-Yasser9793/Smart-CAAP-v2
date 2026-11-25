"""Clinical calibration metrics, symmetric temperature scaling, exact DeLong,
McNemar's test, patient-cluster bootstrap 95% CIs, and IEEE LaTeX formatters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Numerically stable sigmoid."""
    arr = np.asarray(x, dtype=np.float64)
    return np.where(arr >= 0, 1.0 / (1.0 + np.exp(-arr)), np.exp(arr) / (1.0 + np.exp(arr)))


def compute_ece_equal_mass(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Equal-mass quantile Expected Calibration Error (matching Notebook Cells 15/17/18)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    order = np.argsort(p)
    p_s, y_s = p[order], y_t[order]
    bins = np.array_split(np.arange(len(p_s)), n_bins)
    ece = 0.0
    n = len(p_s)
    for b in bins:
        if len(b) > 0:
            ece += (len(b) / n) * abs(float(y_s[b].mean()) - float(p_s[b].mean()))
    return float(ece)


def compute_nll(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Binary negative log-likelihood (log-loss)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y_t * np.log(p) + (1.0 - y_t) * np.log(1.0 - p)))


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score."""
    return float(brier_score_loss(np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=np.float64)))


def fit_temperature_scaling(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Fit post-hoc temperature T* on validation logits via L-BFGS-B minimizing NLL.

    Matches Notebook Cell 15 exactly: x0=[1.0], bounds=[(0.2, 3.0)], method='L-BFGS-B'.
    """
    logits = np.asarray(val_logits, dtype=np.float64)
    labels = np.asarray(val_labels, dtype=np.float64)

    def nll_loss(t: np.ndarray) -> float:
        t_val = max(float(t[0]), 0.05)
        probs = np.clip(sigmoid(logits / t_val), 1e-7, 1.0 - 1e-7)
        return float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))

    res = opt.minimize(nll_loss, x0=[1.0], bounds=[(0.2, 3.0)], method="L-BFGS-B")
    return float(res.x[0])


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    n_bins: int = 10,
) -> dict[str, float | int]:
    """Compute discrimination, calibration, RPI, PPV/NPV, and exact integer TP/TN/FP/FN."""
    y_t = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=np.float64)
    y_pred = (p >= threshold).astype(int)

    auc = float(roc_auc_score(y_t, p))
    ece = compute_ece_equal_mass(y_t, p, n_bins=n_bins)
    brier = compute_brier(y_t, p)
    nll = compute_nll(y_t, p)
    rpi = auc * (1.0 - ece) * 100.0

    tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    f1 = float(f1_score(y_t, y_pred, zero_division=0))
    acc = float(accuracy_score(y_t, y_pred))

    return {
        "ROC-AUC": auc,
        "ECE": ece,
        "Brier": brier,
        "NLL": nll,
        "RPI (%)": rpi,
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV": ppv,
        "NPV": npv,
        "F1-Score": f1,
        "Accuracy": acc,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


# ---------------------------------------------------------------------------
# Fast Exact DeLong Test (Sun & Xu, IEEE Signal Processing Letters, 2014)
# ---------------------------------------------------------------------------
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1.0
        i = k
    out = np.empty(n, dtype=np.float64)
    out[j] = t
    return out


def _fast_delong(preds_sorted_transposed: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    k = preds_sorted_transposed.shape[0]
    pos = preds_sorted_transposed[:, :m]
    neg = preds_sorted_transposed[:, m:]

    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - float(m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_roc_test(y_true: np.ndarray, y_prob_a: np.ndarray, y_prob_b: np.ndarray) -> dict[str, float]:
    """Two-sided correlated DeLong test comparing ROC-AUC(A) vs ROC-AUC(B)."""
    y_t = np.asarray(y_true, dtype=int)
    p_a = np.asarray(y_prob_a, dtype=np.float64)
    p_b = np.asarray(y_prob_b, dtype=np.float64)

    order = np.argsort(-y_t)
    y_sorted = y_t[order]
    label_1_count = int(y_sorted.sum())
    preds = np.vstack([p_a[order], p_b[order]])

    aucs, cov = _fast_delong(preds, label_1_count)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var <= 1e-15:
        z_stat, p_val = 0.0, 1.0
    else:
        z_stat = diff / np.sqrt(var)
        p_val = float(2.0 * stats.norm.sf(abs(z_stat)))

    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "delta_auc": diff,
        "z_stat": float(z_stat),
        "p_value": p_val,
    }


def mcnemar_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, float | int | str]:
    """Two-sided McNemar test on paired binary predictions (exact if discordant < 25)."""
    y_t = np.asarray(y_true, dtype=int)
    correct_a = (np.asarray(y_prob_a, dtype=np.float64) >= threshold).astype(int) == y_t
    correct_b = (np.asarray(y_prob_b, dtype=np.float64) >= threshold).astype(int) == y_t

    b = int(np.sum(correct_a & (~correct_b)))  # A right, B wrong
    c = int(np.sum((~correct_a) & correct_b))  # A wrong, B right
    n_disc = b + c

    if n_disc == 0:
        return {"b_a_right_b_wrong": b, "c_a_wrong_b_right": c, "stat": 0.0, "p_value": 1.0, "method": "exact"}
    if n_disc < 25:
        res = stats.binomtest(min(b, c), n=n_disc, p=0.5, alternative="two-sided")
        return {
            "b_a_right_b_wrong": b,
            "c_a_wrong_b_right": c,
            "stat": float(min(b, c)),
            "p_value": float(res.pvalue),
            "method": "exact_binomial",
        }
    chi2 = ((abs(b - c) - 1.0) ** 2) / float(n_disc)
    p_val = float(stats.chi2.sf(chi2, df=1))
    return {
        "b_a_right_b_wrong": b,
        "c_a_wrong_b_right": c,
        "stat": float(chi2),
        "p_value": p_val,
        "method": "chi2_continuity_corrected",
    }


# ---------------------------------------------------------------------------
# Patient-Cluster Bootstrap (B = 1,000, seed = 42)
# ---------------------------------------------------------------------------
def patient_cluster_bootstrap_ci(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Compute patient-clustered 95% percentile bootstrap CIs for all key metrics."""
    rng = np.random.RandomState(seed)
    pids = np.asarray(patient_ids)
    y_t = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=np.float64)

    unique_pids = np.unique(pids)
    pid_to_idx = {pid: np.where(pids == pid)[0] for pid in unique_pids}

    keys = ["ROC-AUC", "ECE", "Brier", "NLL", "RPI (%)", "Sensitivity", "Specificity", "PPV", "NPV", "F1-Score", "Accuracy"]
    samples: dict[str, list[float]] = {k: [] for k in keys}

    for _ in range(n_boot):
        sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        idx = np.concatenate([pid_to_idx[pid] for pid in sampled])
        if len(np.unique(y_t[idx])) < 2:
            continue
        m = compute_clinical_metrics(y_t[idx], p[idx], threshold=threshold)
        for k in keys:
            samples[k].append(float(m[k]))

    return {
        k: (float(np.percentile(samples[k], 2.5)), float(np.percentile(samples[k], 97.5)))
        for k in keys
    }


def paired_patient_cluster_bootstrap(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    threshold: float = 0.50,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Paired patient-cluster bootstrap comparing Pipeline A (CAAP) vs Pipeline B (Baseline).

    Sign convention: positive delta favors Pipeline A:
      - delta_auc   = AUC(A) - AUC(B)
      - delta_ece   = ECE(B) - ECE(A)   (positive = lower ECE for A)
      - delta_brier = Brier(B) - Brier(A)
      - delta_nll   = NLL(B) - NLL(A)
      - delta_rpi   = RPI(A) - RPI(B)
    """
    rng = np.random.RandomState(seed)
    pids = np.asarray(patient_ids)
    y_t = np.asarray(y_true, dtype=int)
    p_a = np.asarray(y_prob_a, dtype=np.float64)
    p_b = np.asarray(y_prob_b, dtype=np.float64)

    unique_pids = np.unique(pids)
    pid_to_idx = {pid: np.where(pids == pid)[0] for pid in unique_pids}

    diffs: dict[str, list[float]] = {
        "delta_auc": [],
        "delta_ece": [],
        "delta_brier": [],
        "delta_nll": [],
        "delta_rpi": [],
    }
    for _ in range(n_boot):
        sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        idx = np.concatenate([pid_to_idx[pid] for pid in sampled])
        if len(np.unique(y_t[idx])) < 2:
            continue
        ma = compute_clinical_metrics(y_t[idx], p_a[idx], threshold=threshold)
        mb = compute_clinical_metrics(y_t[idx], p_b[idx], threshold=threshold)
        diffs["delta_auc"].append(float(ma["ROC-AUC"]) - float(mb["ROC-AUC"]))
        diffs["delta_ece"].append(float(mb["ECE"]) - float(ma["ECE"]))
        diffs["delta_brier"].append(float(mb["Brier"]) - float(ma["Brier"]))
        diffs["delta_nll"].append(float(mb["NLL"]) - float(ma["NLL"]))
        diffs["delta_rpi"].append(float(ma["RPI (%)"]) - float(mb["RPI (%)"]))

    summary: dict[str, dict[str, float]] = {}
    for k, vals in diffs.items():
        arr = np.asarray(vals, dtype=np.float64)
        p_two = float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0)))
        summary[k] = {
            "mean_diff": float(arr.mean()),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
            "p_two_sided": min(p_two, 1.0),
        }
    return summary


def format_ieee_metrics_latex(
    df_rows: pd.DataFrame,
    caption: str,
    label: str,
) -> str:
    """Format benchmark rows (with 95% CIs, PPV/NPV, and integer TP/TN/FP/FN) for IEEEtran."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccccc}",
        r"\toprule",
        r"Pipeline ($T$) & ROC-AUC (95\% CI) & ECE $\downarrow$ (95\% CI) & Brier $\downarrow$ & NLL $\downarrow$ & RPI (\%) $\uparrow$ & Sens. & Spec. & PPV & NPV & TP / TN / FP / FN \\",
        r"\midrule",
    ]
    for _, r in df_rows.iterrows():
        name_tex = str(r["Pipeline"]).replace("_", r"\_")
        lines.append(
            f"{name_tex} ($T={r['T']:.4f}$) & "
            f"{r['ROC-AUC']:.4f} ({r['AUC_CI_L']:.4f}--{r['AUC_CI_H']:.4f}) & "
            f"{r['ECE']:.4f} ({r['ECE_CI_L']:.4f}--{r['ECE_CI_H']:.4f}) & "
            f"{r['Brier']:.4f} & {r['NLL']:.4f} & {r['RPI (%)']:.2f} & "
            f"{r['Sensitivity']:.4f} & {r['Specificity']:.4f} & {r['PPV']:.4f} & {r['NPV']:.4f} & "
            f"{int(r['TP'])}/{int(r['TN'])}/{int(r['FP'])}/{int(r['FN'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"])
    return "\n".join(lines)
