#!/usr/bin/env python3
"""Render side-by-side crops for Xnemo conditioning runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw

from gaussianavatars.utils.region_loss_utils import RegionMaskProjector, region_l1_metrics


def assert_cuda_or_fail() -> None:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return
    report = {
        "status": "failed",
        "stage": "cuda_precheck",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_is_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "reason": "CUDA is required for rendering.",
    }
    print(json.dumps(report, indent=2), file=sys.stderr)
    raise SystemExit(2)


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().float().cpu().clamp(0.0, 1.0)
    while image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    elif image.ndim == 3:
        image = image.permute(1, 2, 0)
    elif image.ndim != 2:
        raise ValueError(f"Cannot save tensor image with shape {tuple(tensor.shape)}")
    array = (image.numpy() * 255.0).round().astype("uint8")
    Image.fromarray(array).save(path)


def save_crops(image_path: Path, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    boxes = {
        "brow": (int(0.16 * w), int(0.10 * h), int(0.84 * w), int(0.38 * h)),
        "eyes": (int(0.12 * w), int(0.20 * h), int(0.88 * w), int(0.56 * h)),
        "mouth": (int(0.20 * w), int(0.55 * h), int(0.80 * w), int(0.92 * h)),
    }
    outputs = {}
    for name, box in boxes.items():
        out_path = out_dir / f"{name}.png"
        image.crop(box).resize((192, 192), Image.BILINEAR).save(out_path)
        outputs[name] = out_path
    return outputs


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((pred - gt) ** 2).clamp_min(1e-10)
    return float((-10.0 * torch.log10(mse)).detach().cpu())


def parse_run_specs(specs: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Bad --run spec {spec!r}. Use NAME=PATH.")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        path = Path(raw_path.strip())
        if not name or not str(path):
            raise ValueError(f"Bad --run spec {spec!r}. Use NAME=PATH.")
        runs[name] = path
    return runs


def all_source_cameras(scene, split: str):
    groups = []
    if split in ("all", "train"):
        groups.append(scene.getTrainCameras())
    if split in ("all", "val"):
        groups.append(scene.getValCameras())
    if split in ("all", "test"):
        groups.append(scene.getTestCameras())

    cameras = []
    seen = set()
    for group in groups:
        for camera in group:
            key = (
                str(getattr(camera, "image_path", "")),
                str(getattr(camera, "image_name", "")),
                int(getattr(camera, "timestep", 0)),
            )
            if key in seen:
                continue
            seen.add(key)
            cameras.append(camera)
    cameras.sort(key=lambda cam: int(getattr(cam, "timestep", 0)))
    return cameras


def select_timesteps(cameras, frame_indices: list[int] | None, max_frames: int) -> list[int]:
    by_timestep = {int(getattr(camera, "timestep", 0)): camera for camera in cameras}
    if frame_indices:
        missing = [idx for idx in frame_indices if idx not in by_timestep]
        if missing:
            raise ValueError(f"Requested timesteps not found in selected split: {missing}")
        return frame_indices
    timesteps = sorted(by_timestep)
    if max_frames <= 0 or len(timesteps) <= max_frames:
        return timesteps
    picks = np.linspace(0, len(timesteps) - 1, max_frames).round().astype(int)
    return [timesteps[int(idx)] for idx in picks]


def make_strip(image_paths: list[Path], labels: list[str], out_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    w = max(image.width for image in images)
    h = max(image.height for image in images)
    label_h = 28
    canvas = Image.new("RGB", (w * len(images), h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (image, label) in enumerate(zip(images, labels)):
        if image.size != (w, h):
            image = image.resize((w, h), Image.BILINEAR)
        x = idx * w
        canvas.paste(image, (x, label_h))
        draw.text((x + 6, 6), label, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def load_run(run_name: str, model_path: Path, source_paths: list[str], iteration: int | None = None):
    from omegaconf import OmegaConf

    from gaussianavatars.scene.cap4d_gaussian_model_xnemo import CAP4DGaussianModel
    from gaussianavatars.scene.scene import Scene
    from gaussianavatars.utils.system_utils import searchForMaxIteration

    config_path = model_path / "config_dump.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_dump.yaml for {run_name}: {config_path}")

    config = OmegaConf.load(config_path)
    gaussians = CAP4DGaussianModel(config["model_params"])
    gaussians.eval()
    scene = Scene(
        source_paths=source_paths,
        model_path=str(model_path),
        gaussians=gaussians,
        shuffle=False,
    )

    if iteration is None:
        loaded_iter, chkpt_path = searchForMaxIteration(model_path)
        if loaded_iter is None:
            raise FileNotFoundError(f"No checkpoint found for {run_name}: {model_path}")
        chkpt_path = Path(chkpt_path)
    else:
        loaded_iter = int(iteration)
        chkpt_path = model_path / f"chkpnt{loaded_iter}.pth"
        if not chkpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {run_name}: {chkpt_path}")
    model_weights, _ = torch.load(chkpt_path, weights_only=False)
    gaussians.restore(model_weights)
    gaussians.eval()
    return config, scene, gaussians, int(loaded_iter), Path(chkpt_path)


def render_run(
    run_name: str,
    model_path: Path,
    source_paths: list[str],
    timesteps: list[int],
    split: str,
    out_root: Path,
    iteration: int | None = None,
):
    from gaussianavatars.gaussian_renderer.gsplat_renderer import render

    config, scene, gaussians, iteration, chkpt_path = load_run(run_name, model_path, source_paths, iteration)
    cameras = all_source_cameras(scene, split)
    cameras_by_timestep = {int(getattr(camera, "timestep", 0)): camera for camera in cameras}
    region_projector = RegionMaskProjector(gaussians.flame_verts)
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    records = []
    for timestep in timesteps:
        camera = cameras_by_timestep[int(timestep)]
        if gaussians.binding is not None:
            gaussians.select_mesh_by_timestep(camera.timestep)
        render_pkg = render(camera, gaussians, background)
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
        err = torch.abs(image - gt)
        region_masks = region_projector.build_masks(gaussians.current_flame_verts_for_region_masks, camera)
        region_metrics = region_l1_metrics(image, gt, region_masks)

        stem = str(getattr(camera, "image_name", f"{timestep:05d}")).split(".")[0]
        frame_dir = out_root / f"frame_{int(timestep):05d}_{stem}"
        run_dir = frame_dir / run_name
        render_path = run_dir / "render.png"
        error_path = run_dir / "error.png"
        save_tensor_image(image, render_path)
        save_tensor_image(err / err.max().clamp_min(1e-8), error_path)
        save_crops(render_path, run_dir / "render_crops")
        save_crops(error_path, run_dir / "error_crops")

        gt_path = frame_dir / "gt.png"
        if not gt_path.exists():
            save_tensor_image(gt, gt_path)
            save_crops(gt_path, frame_dir / "gt_crops")

        records.append(
            {
                "run": run_name,
                "iteration": iteration,
                "checkpoint": str(chkpt_path),
                "timestep": int(timestep),
                "stem": stem,
                "l1": float(err.mean().detach().cpu()),
                "psnr": psnr(image, gt),
                "region_full_face_l1": region_metrics.get("region/full_face_l1"),
                "region_mouth_l1": region_metrics.get("region/mouth_l1"),
                "region_eyes_l1": region_metrics.get("region/eyes_l1"),
                "region_brow_l1": region_metrics.get("region/brow_l1"),
                "motion_condition_mode": config["model_params"].get("motion_condition_mode"),
                "motion_feature_path": config["model_params"].get("motion_feature_path"),
            }
        )
    return records


def write_comparison_strips(out_root: Path, timesteps: list[int], run_names: list[str]) -> None:
    for frame_dir in sorted(out_root.glob("frame_*")):
        gt_path = frame_dir / "gt.png"
        if not gt_path.exists():
            continue
        image_paths = [gt_path]
        labels = ["gt"]
        for run_name in run_names:
            path = frame_dir / run_name / "render.png"
            if path.exists():
                image_paths.append(path)
                labels.append(run_name)
        make_strip(image_paths, labels, frame_dir / "compare_render.png")

        for region in ("mouth", "eyes", "brow"):
            crop_paths = [frame_dir / "gt_crops" / f"{region}.png"]
            crop_labels = ["gt"]
            for run_name in run_names:
                path = frame_dir / run_name / "render_crops" / f"{region}.png"
                if path.exists():
                    crop_paths.append(path)
                    crop_labels.append(run_name)
            if all(path.exists() for path in crop_paths):
                make_strip(crop_paths, crop_labels, frame_dir / f"compare_{region}.png")


def summarize(records: list[dict]) -> dict:
    by_run: dict[str, list[dict]] = {}
    for record in records:
        by_run.setdefault(record["run"], []).append(record)
    summary = {}
    keys = ("l1", "psnr", "region_full_face_l1", "region_mouth_l1", "region_eyes_l1", "region_brow_l1")
    for run_name, items in by_run.items():
        summary[run_name] = {
            "num_frames": len(items),
            **{
                f"mean_{key}": float(np.mean([item[key] for item in items if item.get(key) is not None]))
                for key in keys
                if any(item.get(key) is not None for item in items)
            },
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_paths", nargs="+", required=True)
    parser.add_argument("--run", action="append", required=True, help="NAME=PATH. May be repeated.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    parser.add_argument("--max_frames", type=int, default=10)
    parser.add_argument("--frame_indices", nargs="*", type=int, default=None)
    parser.add_argument("--iteration", type=int, default=None, help="Checkpoint iteration to render. Defaults to max.")
    args = parser.parse_args()

    assert_cuda_or_fail()
    runs = parse_run_specs(args.run)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    first_config, first_scene, _first_gaussians, _iteration, _chkpt = load_run(
        next(iter(runs.keys())),
        next(iter(runs.values())),
        args.source_paths,
        args.iteration,
    )
    cameras = all_source_cameras(first_scene, args.split)
    timesteps = select_timesteps(cameras, args.frame_indices, args.max_frames)

    records: list[dict] = []
    for run_name, model_path in runs.items():
        records.extend(
            render_run(
                run_name,
                model_path,
                args.source_paths,
                timesteps,
                args.split,
                out_root,
                args.iteration,
            )
        )
    write_comparison_strips(out_root, timesteps, list(runs.keys()))

    payload = {
        "source_paths": args.source_paths,
        "split": args.split,
        "timesteps": timesteps,
        "runs": {name: str(path) for name, path in runs.items()},
        "summary": summarize(records),
        "frames": records,
    }
    (out_root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote renders to {out_root}")


if __name__ == "__main__":
    main()
