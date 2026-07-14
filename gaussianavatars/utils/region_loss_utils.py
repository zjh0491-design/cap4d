from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter


REGION_ORDER = ("mouth", "brow", "eyes", "cheeks")


@dataclass(frozen=True)
class RegionMaskConfig:
    feather_px: float = 3.0
    mouth_scale: float = 1.18
    eyes_scale: float = 1.20
    brow_scale: float = 1.18
    cheeks_scale: float = 1.08
    min_valid_pixels: float = 8.0


def _convex_hull(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] <= 2:
        return points
    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float32)
    return hull if hull.shape[0] >= 3 else points


def _expand_polygon(points: np.ndarray, scale: float, min_pad_px: float = 2.0) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    center = points.mean(axis=0, keepdims=True)
    vec = points - center
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    direction = vec / np.maximum(norm, 1e-6)
    return center + vec * scale + direction * min_pad_px


def _region_components(template_vertices: np.ndarray) -> Dict[str, List[np.ndarray]]:
    v = template_vertices

    mouth = (
        (np.abs(v[:, 0]) < 0.075)
        & (v[:, 1] > -0.165)
        & (v[:, 1] < -0.025)
        & (v[:, 2] > -0.090)
        & (v[:, 2] < 0.075)
    )

    eye_base = (
        (np.abs(v[:, 0]) > 0.018)
        & (np.abs(v[:, 0]) < 0.095)
        & (v[:, 1] > 0.010)
        & (v[:, 1] < 0.085)
        & (v[:, 2] > -0.085)
        & (v[:, 2] < 0.065)
    )
    brow_base = (
        (np.abs(v[:, 0]) > 0.018)
        & (np.abs(v[:, 0]) < 0.095)
        & (v[:, 1] >= 0.075)
        & (v[:, 1] < 0.135)
        & (v[:, 2] > -0.085)
        & (v[:, 2] < 0.065)
    )
    cheek_base = (
        (np.abs(v[:, 0]) > 0.040)
        & (np.abs(v[:, 0]) < 0.105)
        & (v[:, 1] > -0.115)
        & (v[:, 1] < 0.025)
        & (v[:, 2] > -0.040)
        & (v[:, 2] < 0.076)
    )

    return {
        "mouth": [np.flatnonzero(mouth)],
        "eyes": [
            np.flatnonzero(eye_base & (v[:, 0] < -0.018)),
            np.flatnonzero(eye_base & (v[:, 0] > 0.018)),
        ],
        "brow": [
            np.flatnonzero(brow_base & (v[:, 0] < -0.018)),
            np.flatnonzero(brow_base & (v[:, 0] > 0.018)),
        ],
        "cheeks": [
            np.flatnonzero(cheek_base & (v[:, 0] < -0.040)),
            np.flatnonzero(cheek_base & (v[:, 0] > 0.040)),
        ],
    }


class RegionMaskProjector:
    """Project GT FLAME template regions into the current cropped training view."""

    def __init__(self, template_vertices: torch.Tensor, config: RegionMaskConfig | None = None):
        self.config = config or RegionMaskConfig()
        template_np = template_vertices.detach().float().cpu().numpy()
        self.region_components = _region_components(template_np)
        self.region_vertex_counts = {
            name: int(sum(component.size for component in components))
            for name, components in self.region_components.items()
        }

    def _project_vertices(self, vertices: torch.Tensor, camera) -> Tuple[np.ndarray, np.ndarray]:
        if vertices.ndim == 3:
            vertices = vertices[0]
        device = vertices.device
        rt = camera.rt.to(device=device, dtype=vertices.dtype)
        intr = camera.intrinsics.to(device=device, dtype=vertices.dtype)
        verts_cam = vertices @ rt[:3, :3].T + rt[:3, 3]
        z = verts_cam[:, 2]
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        xy = torch.stack(
            [
                verts_cam[:, 0] / z.clamp_min(1e-6) * fx + cx,
                verts_cam[:, 1] / z.clamp_min(1e-6) * fy + cy,
            ],
            dim=-1,
        )
        return xy.detach().cpu().numpy(), z.detach().cpu().numpy()

    def _draw_component(
        self,
        canvas: Image.Image,
        points: np.ndarray,
        region_name: str,
    ) -> None:
        if points.shape[0] < 3:
            return
        scale = {
            "mouth": self.config.mouth_scale,
            "eyes": self.config.eyes_scale,
            "brow": self.config.brow_scale,
            "cheeks": self.config.cheeks_scale,
        }[region_name]
        hull = _convex_hull(points)
        poly = _expand_polygon(hull, scale=scale)
        ImageDraw.Draw(canvas).polygon([tuple(p) for p in poly], fill=255)

    def build_masks(self, vertices: torch.Tensor, camera) -> Dict[str, torch.Tensor]:
        height = int(camera.image_height)
        width = int(camera.image_width)
        xy, z = self._project_vertices(vertices.detach(), camera)
        masks: Dict[str, torch.Tensor] = {}
        used = torch.zeros((1, height, width), dtype=torch.float32, device=vertices.device)
        crop_mask = None
        if getattr(camera, "mask", None) is not None:
            crop_mask = camera.mask.to(device=vertices.device, dtype=torch.float32)[None]

        for region_name in REGION_ORDER:
            canvas = Image.new("L", (width, height), 0)
            for component in self.region_components[region_name]:
                if component.size == 0:
                    continue
                pts = xy[component]
                depth = z[component]
                valid = (
                    np.isfinite(pts).all(axis=1)
                    & np.isfinite(depth)
                    & (depth > 1e-5)
                    & (pts[:, 0] > -width)
                    & (pts[:, 0] < 2 * width)
                    & (pts[:, 1] > -height)
                    & (pts[:, 1] < 2 * height)
                )
                self._draw_component(canvas, pts[valid], region_name)
            if self.config.feather_px > 0:
                canvas = canvas.filter(ImageFilter.GaussianBlur(radius=self.config.feather_px))
            mask_np = np.asarray(canvas, dtype=np.float32) / 255.0
            mask = torch.from_numpy(mask_np)[None].to(device=vertices.device, dtype=torch.float32)
            if crop_mask is not None:
                mask = mask * crop_mask
            mask = mask * (1.0 - used).clamp_min(0.0)
            mask = mask.clamp(0.0, 1.0)
            if float(mask.sum().detach().cpu()) < self.config.min_valid_pixels:
                mask = torch.zeros_like(mask)
            masks[region_name] = mask
            used = (used + mask).clamp(0.0, 1.0)
        return masks

    def build_uv_masks(
        self,
        flame_faces: torch.Tensor,
        pix_to_face: torch.Tensor,
        uv_mask: torch.Tensor | None = None,
        target_size: tuple[int, int] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Rasterize the same semantic regions into UV/deformation-map space."""
        device = flame_faces.device
        if pix_to_face.ndim == 4 and pix_to_face.shape[-1] == 1:
            face_ids = pix_to_face[..., 0]
        elif pix_to_face.ndim == 4 and pix_to_face.shape[1] == 1:
            face_ids = pix_to_face[:, 0]
        elif pix_to_face.ndim == 3:
            face_ids = pix_to_face
        else:
            raise ValueError(f"Unsupported pix_to_face shape: {tuple(pix_to_face.shape)}")

        face_ids = face_ids.to(device=device)
        if uv_mask is None:
            valid = face_ids >= 0
        else:
            if uv_mask.ndim == 4:
                valid = uv_mask[:, 0].to(device=device).bool()
            elif uv_mask.ndim == 3:
                valid = uv_mask.to(device=device).bool()
            else:
                raise ValueError(f"Unsupported uv_mask shape: {tuple(uv_mask.shape)}")

        face_ids = face_ids.long().clamp_min(0)
        num_vertices = int(flame_faces.max().detach().cpu().item()) + 1
        masks: Dict[str, torch.Tensor] = {}
        used = torch.zeros_like(face_ids, dtype=torch.float32, device=device)
        for region_name in REGION_ORDER:
            vertex_mask = torch.zeros(num_vertices, dtype=torch.bool, device=device)
            for component in self.region_components[region_name]:
                if component.size == 0:
                    continue
                component_idx = torch.as_tensor(component, dtype=torch.long, device=device)
                component_idx = component_idx[(component_idx >= 0) & (component_idx < num_vertices)]
                vertex_mask[component_idx] = True
            face_mask = vertex_mask[flame_faces.long()].any(dim=-1)
            mask = face_mask[face_ids].to(dtype=torch.float32) * valid.to(dtype=torch.float32)
            mask = mask * (1.0 - used).clamp_min(0.0)
            mask = mask.clamp(0.0, 1.0)
            used = (used + mask).clamp(0.0, 1.0)
            if target_size is not None and tuple(mask.shape[-2:]) != tuple(target_size):
                mask = F.interpolate(
                    mask[:, None],
                    size=target_size,
                    mode="nearest",
                )[:, 0]
            masks[region_name] = mask
        return masks


def charbonnier(diff: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(diff * diff + eps * eps)


def masked_region_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "charbonnier",
) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0) * pred.shape[0]
    diff = pred - target
    if loss_type == "l1":
        per_pixel = diff.abs()
    elif loss_type == "charbonnier":
        per_pixel = charbonnier(diff)
    else:
        raise ValueError(f"Unsupported region loss type: {loss_type}")
    return (per_pixel * mask).sum() / denom


def region_l1_metrics(pred: torch.Tensor, target: torch.Tensor, masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
    metrics = {}
    for name, mask in masks.items():
        if float(mask.sum().detach().cpu()) <= 0:
            metrics[f"region/{name}_l1"] = 0.0
            metrics[f"region/{name}_mask_pixels"] = 0.0
            continue
        metrics[f"region/{name}_l1"] = float(masked_region_loss(pred, target, mask, "l1").detach().cpu())
        metrics[f"region/{name}_mask_pixels"] = float(mask.sum().detach().cpu())
    metrics["region/full_face_l1"] = float((pred - target).abs().mean().detach().cpu())
    overlap = torch.zeros_like(next(iter(masks.values()))) if masks else None
    if overlap is not None:
        raw_sum = sum(masks.values())
        metrics["region/mask_overlap_soft_pixels"] = float((raw_sum - raw_sum.clamp(0.0, 1.0)).sum().detach().cpu())
    return metrics
