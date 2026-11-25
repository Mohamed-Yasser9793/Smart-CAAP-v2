"""ROI-anchored CAAP signal path from the research notebook.

The arithmetic follows the final router (central 60% ROI, P2–P98 span,
bilateral denoise, residual blend, validation thresholds). Population
metrics are not computed here.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

IMG_SIZE = 320
CROP_FRAC = 0.60

# Validation-set calibration recorded in the notebook output:
# TAU_CONTRAST 0.2353 | TAU_NOISE 0.003342 | OPTIMAL_T 1.2215
TAU_CONTRAST_CAL = 0.2353
TAU_NOISE_CAL = 0.003342
OPTIMAL_T_CAL = 1.2215

# Operating points used by the study. Full-test tables use 0.50.
# YOUDEN_T3 is the constant stored in the notebook for ER-triage mode.
THRESHOLD_STANDARD = 0.50
YOUDEN_T3 = 0.330

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

PATH_BYPASS = "Path 1: Clean bypass"
PATH_STRETCH = "Path 2: ROI-anchored soft stretch"
PATH_DENOISE = "Path 3: Bilateral denoise + stretch"


def prepare_radiograph(image: Image.Image, size: int = IMG_SIZE) -> np.ndarray:
    """Grayscale, then area-resize to the training resolution."""
    gray = np.array(image.convert("L"))
    if gray.ndim != 2:
        raise ValueError("Expected a single-channel radiograph after grayscale conversion.")
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def get_central_roi(img_u8: np.ndarray, crop_frac: float = CROP_FRAC):
    h, w = img_u8.shape
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img_u8[y0 : y0 + ch, x0 : x0 + cw], (x0, y0, cw, ch)


def measure_roi_sensors(img_u8: np.ndarray, crop_frac: float = CROP_FRAC):
    """Robust contrast span (P98 − P2) and Laplacian energy inside the ROI."""
    roi_u8, _ = get_central_roi(img_u8, crop_frac)
    roi01 = roi_u8.astype(np.float32) / 255.0
    p2, p98 = np.percentile(roi01, [2, 98])
    c_span = float(p98 - p2)
    hf_energy = float(np.var(cv2.Laplacian(roi01, cv2.CV_32F)))
    return c_span, hf_energy


def detect_metallic_implant(img_u8: np.ndarray, intensity_thresh: float = 0.95, min_area_ratio: float = 0.002):
    """Bright, sharp regions used in the notebook as an orthopedic-hardware proxy."""
    img01 = img_u8.astype(np.float32) / 255.0
    bright_mask = (img01 >= intensity_thresh).astype(np.uint8)
    bright_ratio = float(bright_mask.mean())
    if bright_ratio < min_area_ratio:
        return False, bright_ratio, 0.0
    sobelx = cv2.Sobel(img01, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img01, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    kernel = np.ones((5, 5), np.uint8)
    boundary_mask = cv2.dilate(bright_mask, kernel, iterations=1) - bright_mask
    edge_score = float(grad_mag[boundary_mask > 0].mean()) if boundary_mask.sum() > 0 else 0.0
    is_hardware = (bright_ratio >= min_area_ratio) and (edge_score >= 0.15)
    return bool(is_hardware), bright_ratio, edge_score


def simulate_degradation(img_u8: np.ndarray, alpha: float = 1.0, sigma: float = 0.0, seed: int = 42) -> np.ndarray:
    """Linear contrast compression about the mean, then seeded Gaussian sensor noise.

    Matches the notebook signal cell. Alpha = 1 and sigma = 0 return the input
    unchanged. Noise uses a fixed seed so Streamlit reruns stay stable.
    """
    if alpha == 1.0 and sigma <= 0.0:
        return img_u8
    img01 = img_u8.astype(np.float32) / 255.0
    mu = float(img01.mean())
    faded01 = np.clip(alpha * (img01 - mu) + mu, 0.0, 1.0)
    faded = np.rint(faded01 * 255.0).astype(np.float32)
    if sigma > 0.0:
        noise = np.random.default_rng(seed).normal(0.0, sigma, size=img_u8.shape).astype(np.float32)
        faded = faded + noise
    return np.clip(np.rint(faded), 0, 255).astype(np.uint8)


def apply_clahe(img_u8: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_u8)


def apply_global_blind_stretch(img_u8: np.ndarray) -> np.ndarray:
    img01 = img_u8.astype(np.float32) / 255.0
    p1, p99 = np.percentile(img01, [1, 99])
    stretched = np.clip((img01 - p1) / max(float(p99 - p1), 1e-5), 0.0, 1.0)
    return np.rint(stretched * 255.0).astype(np.uint8)


def apply_roi_anchored_stretch(img_u8: np.ndarray, crop_frac: float = CROP_FRAC, alpha_blend: float = 0.75) -> np.ndarray:
    roi_u8, _ = get_central_roi(img_u8, crop_frac)
    p2, p98 = np.percentile(roi_u8, [2, 98])
    if (p98 - p2) < 10.0:
        p2, p98 = np.percentile(img_u8, [1, 99])
    stretched_u8 = np.clip(
        (img_u8.astype(np.float32) - p2) * 255.0 / max(float(p98 - p2), 1.0),
        0.0,
        255.0,
    )
    blended = (alpha_blend * stretched_u8) + ((1.0 - alpha_blend) * img_u8.astype(np.float32))
    return np.rint(blended).astype(np.uint8)


def apply_edge_preserving_denoise_stretch(img_u8: np.ndarray) -> np.ndarray:
    denoised = cv2.bilateralFilter(img_u8, d=5, sigmaColor=25, sigmaSpace=25)
    return apply_roi_anchored_stretch(denoised, crop_frac=CROP_FRAC, alpha_blend=0.80)


def caap_route_smart(img_u8: np.ndarray):
    """Return the routed image, path name, and the sensors that made the decision."""
    c_span, hf_energy = measure_roi_sensors(img_u8)
    if c_span >= TAU_CONTRAST_CAL and hf_energy <= TAU_NOISE_CAL:
        routed = img_u8
        path = PATH_BYPASS
    elif hf_energy <= TAU_NOISE_CAL:
        routed = apply_roi_anchored_stretch(img_u8)
        path = PATH_STRETCH
    else:
        routed = apply_edge_preserving_denoise_stretch(img_u8)
        path = PATH_DENOISE
    return routed, path, c_span, hf_energy


def bilateral_only(img_u8: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(img_u8, d=5, sigmaColor=25, sigmaSpace=25)


def draw_roi_overlay(img_u8: np.ndarray, crop_frac: float = CROP_FRAC) -> np.ndarray:
    rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    _, (x0, y0, cw, ch) = get_central_roi(img_u8, crop_frac)
    cv2.rectangle(rgb, (x0, y0), (x0 + cw - 1, y0 + ch - 1), (96, 196, 255), 2)
    return rgb


def absolute_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)


def colorize_difference(diff_u8: np.ndarray) -> np.ndarray:
    colored = cv2.applyColorMap(diff_u8, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def wipe_compare(before: np.ndarray, after: np.ndarray, fraction: float) -> np.ndarray:
    """Left side is the model input; right side is the CAAP output."""
    fraction = float(np.clip(fraction, 0.0, 1.0))
    cut = int(round(before.shape[1] * fraction))
    composite = cv2.cvtColor(before, cv2.COLOR_GRAY2RGB)
    after_rgb = cv2.cvtColor(after, cv2.COLOR_GRAY2RGB)
    if cut < before.shape[1]:
        composite[:, cut:] = after_rgb[:, cut:]
    if 0 < cut < before.shape[1]:
        composite[:, cut : cut + 1] = (110, 196, 255)
    return composite


def synthetic_phantom(size: int = IMG_SIZE, contrast: float = 0.55) -> np.ndarray:
    """Labeled demo image. Not a radiograph and not used for reported metrics."""
    y, x = np.ogrid[:size, :size]
    bone = np.exp(-((x - 168) ** 2) / (2 * 46**2) - ((y - 146) ** 2) / (2 * 78**2))
    radius = np.exp(-((x - 188) ** 2) / (2 * 22**2) - ((y - 118) ** 2) / (2 * 16**2))
    ulna = np.exp(-((x - 142) ** 2) / (2 * 16**2) - ((y - 210) ** 2) / (2 * 48**2))
    img = 36.0 + contrast * (bone * 150.0 + radius * 55.0 + ulna * 80.0)
    line = (np.abs(y - 164) <= 1) & (np.abs(x - 168) <= 26)
    img[line] *= 0.72
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)
