"""Grad-CAM failure analysis visualizer for Smart CAAP v2.0 (CLAIM 2024 Items 31 & 39).

Extracts and saves high-resolution (300 DPI) 3-panel side-by-side figures:
  1. Original MURA wrist radiograph (with central 60% ROI box & sensor readouts)
  2. Smart CAAP v2.0 routed radiograph (with active routing path)
  3. ResNet-50 `layer4` Grad-CAM heatmap overlay
for representative False Negatives (FN) and False Positives (FP) on Split B Test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from caap.inference import DEVICE, WEIGHTS_PATH, _batch_tensor, load_backbone, sigmoid
from caap.pipeline import (
    IMG_SIZE,
    OPTIMAL_T_CAL,
    THRESHOLD_STANDARD,
    caap_route_smart,
    detect_metallic_implant,
    draw_roi_overlay,
)
from experiments.data_splits import build_split_b, load_study_images


class ResNet50GradCAM:
    """Standard Grad-CAM extractor attached to ResNet-50 `layer4[-1]`."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer = model.layer4[-1]
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out: tuple[torch.Tensor, ...]) -> None:
        self.gradients = grad_out[0].detach()

    def generate(self, img_u8: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute normalized [0, 1] Grad-CAM heatmap (H, W) and raw logit."""
        self.model.zero_grad(set_to_none=True)
        tensor = _batch_tensor([img_u8]).requires_grad_(True)
        logit = self.model(tensor).view(-1)[0]
        logit.backward()

        assert self.activations is not None and self.gradients is not None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_np = cv2.resize(cam_np, (img_u8.shape[1], img_u8.shape[0]), interpolation=cv2.INTER_CUBIC)
        cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np, dtype=np.float32)
        return cam_np.astype(np.float32), float(logit.detach().cpu().item())

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def overlay_gradcam(img_u8: np.ndarray, cam_01: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    """Blend a [0, 1] Grad-CAM heatmap over a uint8 grayscale radiograph in RGB."""
    base_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB).astype(np.float32)
    heat_bgr = cv2.applyColorMap(np.rint(np.clip(cam_01, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    blended = (1.0 - alpha) * base_rgb + alpha * heat_rgb
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def save_failure_figure(
    raw_u8: np.ndarray,
    routed_u8: np.ndarray,
    cam_01: np.ndarray,
    meta: dict,
    save_path: Path,
) -> None:
    """Save a 300-DPI 3-panel comparison figure for a single FN or FP case."""
    roi_vis = draw_roi_overlay(raw_u8)
    cam_vis = overlay_gradcam(routed_u8, cam_01)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), dpi=300)
    hw_str = "Hardware Flagged" if meta["hardware_flag"] else "No Hardware"
    fig.suptitle(
        f"{meta['error_type']} | Study: {meta['study_id']} | True: {meta['label']} | "
        f"Study Calibrated P(fx)={meta['study_prob']:.3f} (T={meta['temperature']:.4f}) | {hw_str}",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )

    axes[0].imshow(roi_vis)
    axes[0].set_title(
        f"1. Original X-Ray (Central 60% ROI)\nSpan={meta['c_span']:.3f} | HF={meta['hf_energy']:.5f}",
        fontsize=9.5,
        fontweight="bold",
    )
    axes[0].axis("off")

    axes[1].imshow(routed_u8, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(
        f"2. Smart CAAP v2.0 Output\n{meta['route']} | View P={meta['view_prob']:.3f}",
        fontsize=9.5,
        fontweight="bold",
    )
    axes[1].axis("off")

    axes[2].imshow(cam_vis)
    axes[2].set_title(
        "3. Frozen ResNet-50 Grad-CAM (layer4)\nHigh activation = red/yellow",
        fontsize=9.5,
        fontweight="bold",
    )
    axes[2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_gradcam_failure_analysis(
    mura_root: str | Path,
    weights_path: str | Path = WEIGHTS_PATH,
    output_dir: str | Path = "outputs/gradcam_failures",
    top_k: int = 5,
    threshold: float = THRESHOLD_STANDARD,
    temperature: float = OPTIMAL_T_CAL,
) -> pd.DataFrame:
    """Identify top-K highest-confidence FN and FP studies on Split B Test and render Grad-CAM panels."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_b = build_split_b(mura_root)
    test_df: pd.DataFrame = split_b["test"]  # type: ignore[assignment]
    model, _ = load_backbone(Path(weights_path))
    model.to(DEVICE).eval()

    study_records: list[dict] = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            imgs = load_study_images(row["img_paths"], size=IMG_SIZE)
            if not imgs:
                continue
            routed_list, routes, spans, hfs, hw_flags = [], [], [], [], []
            for raw_u8 in imgs:
                r_img, r_name, c_span, hf = caap_route_smart(raw_u8)
                has_hw, _, _ = detect_metallic_implant(raw_u8)
                routed_list.append(r_img)
                routes.append(r_name)
                spans.append(c_span)
                hfs.append(hf)
                hw_flags.append(has_hw)

            logits = model(_batch_tensor(routed_list)).view(-1).cpu().numpy().astype(np.float64)
            mean_logit = float(np.mean(logits))
            study_prob = float(sigmoid(mean_logit / temperature))
            pred = int(study_prob >= threshold)
            label = int(row["label"])

            if label == 1 and pred == 0:
                err_type = "False Negative"
                rep_idx = int(np.argmin(logits))  # Most confident negative view
            elif label == 0 and pred == 1:
                err_type = "False Positive"
                rep_idx = int(np.argmax(logits))  # Most confident positive view
            else:
                continue

            study_records.append(
                {
                    "error_type": err_type,
                    "study_id": row["study_id"],
                    "patient_id": row["patient_id"],
                    "label": label,
                    "study_prob": study_prob,
                    "view_idx": rep_idx,
                    "view_prob": float(sigmoid(logits[rep_idx] / temperature)),
                    "route": routes[rep_idx],
                    "c_span": spans[rep_idx],
                    "hf_energy": hfs[rep_idx],
                    "hardware_flag": any(hw_flags),
                    "temperature": float(temperature),
                    "raw_u8": imgs[rep_idx],
                    "routed_u8": routed_list[rep_idx],
                }
            )

    fn_cases = sorted([r for r in study_records if r["error_type"] == "False Negative"], key=lambda x: x["study_prob"])[
        :top_k
    ]
    fp_cases = sorted([r for r in study_records if r["error_type"] == "False Positive"], key=lambda x: -x["study_prob"])[
        :top_k
    ]
    selected = fn_cases + fp_cases

    cam_extractor = ResNet50GradCAM(model)
    summary_rows: list[dict] = []
    try:
        for rank, case in enumerate(selected, start=1):
            cam_01, _ = cam_extractor.generate(case["routed_u8"])
            prefix = "FN" if case["error_type"] == "False Negative" else "FP"
            fname = f"{prefix}_{rank:02d}_{case['study_id']}.png"
            save_path = out_dir / fname
            save_failure_figure(case["raw_u8"], case["routed_u8"], cam_01, case, save_path)
            summary_rows.append(
                {
                    "error_type": case["error_type"],
                    "study_id": case["study_id"],
                    "patient_id": case["patient_id"],
                    "label": case["label"],
                    "study_prob": round(case["study_prob"], 4),
                    "view_prob": round(case["view_prob"], 4),
                    "route": case["route"],
                    "c_span": round(case["c_span"], 4),
                    "hf_energy": round(case["hf_energy"], 6),
                    "hardware_flag": case["hardware_flag"],
                    "figure_file": str(save_path),
                }
            )
    finally:
        cam_extractor.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "gradcam_failure_cases.csv", index=False)
    print(f"[Grad-CAM] Saved {len(summary_df)} failure figures ({len(fn_cases)} FN, {len(fp_cases)} FP) to {out_dir}")
    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart CAAP v2.0 ResNet-50 Grad-CAM Failure Analysis Visualizer")
    parser.add_argument("--mura-root", type=str, required=True, help="Root path containing Stanford MURA v1.1.")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_PATH), help="Path to best_resnet50_wrist.pth.")
    parser.add_argument("--output-dir", type=str, default="outputs/gradcam_failures", help="Output directory for PNGs.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of representative FN and FP studies to plot.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_STANDARD, help="Decision threshold (default 0.50).")
    parser.add_argument("--temperature", type=float, default=OPTIMAL_T_CAL, help="Calibrated temperature T*.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_gradcam_failure_analysis(
        mura_root=args.mura_root,
        weights_path=args.weights,
        output_dir=args.output_dir,
        top_k=args.top_k,
        threshold=args.threshold,
        temperature=args.temperature,
    )
