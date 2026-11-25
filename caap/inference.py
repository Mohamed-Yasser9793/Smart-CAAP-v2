"""Frozen ResNet-50 inference and live Grad-CAM. Probabilities come from the checkpoint only."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torchvision import models

from caap.pipeline import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    OPTIMAL_T_CAL,
    PATH_DENOISE,
    absolute_difference,
    apply_clahe,
    apply_global_blind_stretch,
    bilateral_only,
    caap_route_smart,
    colorize_difference,
    detect_metallic_implant,
    draw_roi_overlay,
    get_central_roi,
    measure_roi_sensors,
)

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fitted symmetrically via L-BFGS-B on Split B Validation (n = 371 studies)
SYMMETRIC_TEMPS = {
    "Model input": 0.9658,
    "CLAHE": 1.0558,
    "Blind global stretch": 1.2405,
    "CAAP": 0.9460,
}


def resolve_weights_path() -> Path:
    """Locate the frozen ResNet-50 checkpoint in weights/ or repository root."""
    candidates = [
        ROOT / "weights" / "best_resnet50_wrist.pth",
        ROOT / "best_resnet50_wrist.pth",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


WEIGHTS_PATH = resolve_weights_path()


def sigmoid(logit: float) -> float:
    return float(1.0 / (1.0 + np.exp(-logit)))


def load_backbone(weights_path: Path = WEIGHTS_PATH) -> tuple[nn.Module, dict]:
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights_path}. "
            "Ensure Git LFS is pulled (`git lfs pull`) or place best_resnet50_wrist.pth in weights/."
        )
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    n_params = sum(p.numel() for p in model.parameters())
    meta = {
        "path": str(weights_path),
        "filename": weights_path.name,
        "parameters": int(n_params),
        "device": str(DEVICE),
        "fc_out": int(model.fc.out_features),
    }
    return model, meta


def _batch_tensor(images: list[np.ndarray]) -> torch.Tensor:
    arrays = []
    for img_u8 in images:
        x = img_u8.astype(np.float32) / 255.0
        x = np.repeat(x[None, :, :], 3, axis=0)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        arrays.append(torch.from_numpy(np.ascontiguousarray(x)))
    return torch.stack(arrays).to(DEVICE)


def predict_logits(model: nn.Module, images: list[np.ndarray]) -> np.ndarray:
    tensor = _batch_tensor(images)
    with torch.no_grad():
        logits = model(tensor).view(-1).detach().cpu().numpy().astype(np.float64)
    return logits


def compute_live_gradcam(model: nn.Module, img_u8: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    """Compute a ResNet-50 `layer4[-1]` Grad-CAM RGB overlay for a single uint8 radiograph."""
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    target_layer = model.layer4[-1]
    h_fwd = target_layer.register_forward_hook(lambda _m, _i, o: activations.append(o.detach()))
    h_bwd = target_layer.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0].detach()))
    try:
        model.zero_grad(set_to_none=True)
        tensor = _batch_tensor([img_u8]).requires_grad_(True)
        logit = model(tensor).view(-1)[0]
        logit.backward()

        act = activations[0]
        grad = gradients[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_np = cv2.resize(cam_np, (img_u8.shape[1], img_u8.shape[0]), interpolation=cv2.INTER_CUBIC)
        cmin, cmax = float(cam_np.min()), float(cam_np.max())
        cam_01 = (cam_np - cmin) / (cmax - cmin) if (cmax - cmin) > 1e-8 else np.zeros_like(cam_np)
    finally:
        h_fwd.remove()
        h_bwd.remove()
        model.zero_grad(set_to_none=True)

    base_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB).astype(np.float32)
    heat_bgr = cv2.applyColorMap(np.rint(np.clip(cam_01, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    blended = (1.0 - alpha) * base_rgb + alpha * heat_rgb
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def _pair_metrics(reference: np.ndarray, processed: np.ndarray) -> tuple[float | None, float]:
    ssim_val = float(ssim_metric(reference, processed, data_range=255))
    if np.array_equal(reference, processed):
        return None, ssim_val
    return float(psnr_metric(reference, processed, data_range=255)), ssim_val


def analyze_radiograph(
    model: nn.Module,
    img_u8: np.ndarray,
    threshold: float,
    symmetric_calibration: bool = True,
    with_gradcam: bool = False,
) -> dict:
    """Run the four research pipelines and the frozen classifier on one view.

    A single upload is one radiograph. Reported study tables pooled the mean
    logit across views; that reduction is not applied to a one-image upload.
    """
    if img_u8.ndim != 2 or img_u8.dtype != np.uint8:
        raise ValueError("analyze_radiograph expects a uint8 grayscale image.")

    caap_u8, route, c_span, hf_energy = caap_route_smart(img_u8)
    clahe_u8 = apply_clahe(img_u8)
    stretch_u8 = apply_global_blind_stretch(img_u8)
    denoised_u8 = bilateral_only(img_u8) if route == PATH_DENOISE else None
    roi_u8, roi_box = get_central_roi(img_u8)
    has_metal, bright_ratio, edge_score = detect_metallic_implant(img_u8)

    if symmetric_calibration:
        temps = SYMMETRIC_TEMPS
    else:
        temps = {
            "Model input": 1.0,
            "CLAHE": 1.0,
            "Blind global stretch": 1.0,
            "CAAP": OPTIMAL_T_CAL,
        }

    views = [
        ("Model input", img_u8, temps["Model input"], "Unfiltered"),
        ("CLAHE", clahe_u8, temps["CLAHE"], "Fixed 8×8 tiles, clip 2.0"),
        ("Blind global stretch", stretch_u8, temps["Blind global stretch"], "Global P1–P99"),
        ("CAAP", caap_u8, temps["CAAP"], route),
    ]
    logits = predict_logits(model, [image for _, image, _, _ in views])

    pipelines = []
    for (name, image, temperature, note), logit in zip(views, logits):
        raw_probability = sigmoid(float(logit))
        probability = sigmoid(float(logit) / temperature)
        psnr_val, ssim_val = _pair_metrics(img_u8, image)
        post_span, post_hf = measure_roi_sensors(image)
        pipelines.append(
            {
                "name": name,
                "image": image,
                "note": note,
                "logit": float(logit),
                "temperature": float(temperature),
                "raw_probability": raw_probability,
                "probability": probability,
                "decision": "At or above threshold" if probability >= threshold else "Below threshold",
                "psnr": psnr_val,
                "ssim": ssim_val,
                "roi_span": post_span,
                "roi_hf": post_hf,
            }
        )

    diff = absolute_difference(img_u8, caap_u8)
    gradcam_overlay = compute_live_gradcam(model, caap_u8) if with_gradcam else None

    return {
        "input": img_u8,
        "route": route,
        "c_span": c_span,
        "hf_energy": hf_energy,
        "roi": roi_u8,
        "roi_box": roi_box,
        "roi_overlay": draw_roi_overlay(img_u8),
        "clahe": clahe_u8,
        "stretch": stretch_u8,
        "caap": caap_u8,
        "denoised": denoised_u8,
        "difference": diff,
        "difference_color": colorize_difference(diff),
        "gradcam_overlay": gradcam_overlay,
        "hardware_flag": has_metal,
        "bright_ratio": bright_ratio,
        "edge_score": edge_score,
        "pipelines": pipelines,
        "caap_result": pipelines[-1],
        "threshold": float(threshold),
        "symmetric_calibration": bool(symmetric_calibration),
    }
