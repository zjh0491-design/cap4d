#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import os
import random
import shutil
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from PIL import Image

from gaussianavatars.utils.system_utils import searchForMaxIteration
from gaussianavatars.gaussian_renderer.gsplat_renderer import render
from gaussianavatars.scene.cap4d_gaussian_model_xnemo import CAP4DGaussianModel
from gaussianavatars.scene.scene import CameraDataset, Scene
from gaussianavatars.utils.loss_utils import l1_loss, ssim
from gaussianavatars.utils.general_utils import safe_state
from gaussianavatars.utils.image_utils import psnr, error_map
from gaussianavatars.utils.region_loss_utils import (
    RegionMaskConfig,
    RegionMaskProjector,
    masked_region_loss,
    region_l1_metrics,
)
from gaussianavatars.lpipsPyTorch import LPIPS
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def configure_cuda_performance():
    if not torch.cuda.is_available():
        return
    if os.environ.get("CAP4D_DISABLE_TF32", "0") == "1":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    print("Enabled CUDA performance flags: tf32=True cudnn_benchmark=True")


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def next_camera_from_loader(loader, loader_iter):
    try:
        camera = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        camera = next(loader_iter)
    return camera, loader_iter


def validate_source_vector(name, values, source_paths, require_positive_sum=False):
    if values is None:
        return None
    if not source_paths:
        raise ValueError(f"--{name} requires --source_paths.")
    values = [float(value) for value in values]
    if len(values) != len(source_paths):
        raise ValueError(
            f"--{name} expects one value per source path: "
            f"values={len(values)}, source_paths={len(source_paths)}."
        )
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"--{name} values must be finite and non-negative.")
    if require_positive_sum and sum(values) <= 0.0:
        raise ValueError(f"--{name} must contain at least one positive value.")
    return values


def build_source_sampling_weights(cameras, source_sampling_probabilities):
    if source_sampling_probabilities is None:
        return None, None, None

    probabilities = torch.as_tensor(source_sampling_probabilities, dtype=torch.float64)
    if probabilities.ndim != 1 or probabilities.numel() == 0:
        raise ValueError("source_sampling_probabilities must be a non-empty 1D sequence.")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("source_sampling_probabilities must be finite and non-negative.")
    if probabilities.sum() <= 0:
        raise ValueError("source_sampling_probabilities must contain a positive value.")
    probabilities = probabilities / probabilities.sum()
    counts = torch.zeros(len(probabilities), dtype=torch.long)
    source_ids = []
    for camera in cameras:
        source_id = getattr(camera, "source_id", None)
        if source_id is None or not 0 <= int(source_id) < len(probabilities):
            raise ValueError(
                "Source-balanced sampling requires every real camera to have a valid source_id; "
                f"got source_id={source_id!r}."
            )
        source_id = int(source_id)
        source_ids.append(source_id)
        counts[source_id] += 1

    missing = [
        idx for idx, (probability, count) in enumerate(zip(probabilities, counts))
        if probability > 0 and count == 0
    ]
    if missing:
        raise ValueError(
            "Positive source sampling probability was assigned to a source with no training cameras: "
            + ", ".join(str(idx) for idx in missing)
        )

    weights = torch.tensor(
        [float(probabilities[source_id] / counts[source_id]) for source_id in source_ids],
        dtype=torch.float64,
    )
    return weights, probabilities, counts


def make_camera_loader(cameras, source_sampling_probabilities, seed, num_workers):
    dataset = CameraDataset(cameras)
    weights, probabilities, counts = build_source_sampling_weights(
        cameras,
        source_sampling_probabilities,
    )
    loader_kwargs = {
        "batch_size": None,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if weights is None:
        loader = DataLoader(dataset, shuffle=True, **loader_kwargs)
    else:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        loader = DataLoader(dataset, sampler=sampler, **loader_kwargs)
    return loader, probabilities, counts


def source_value_for_camera(camera, values, default=1.0):
    if values is None or getattr(camera, "is_pseudo", False):
        return float(default)
    source_id = getattr(camera, "source_id", None)
    if source_id is None or not 0 <= int(source_id) < len(values):
        raise ValueError(f"Camera has invalid source_id={source_id!r} for source-specific values.")
    return float(values[int(source_id)])


def masked_l1_loss(network_output, gt, mask):
    denom = mask.sum().clamp_min(1.) * network_output.shape[0]
    return (torch.abs(network_output - gt) * mask).sum() / denom


def make_region_projector(gaussians, opt_params):
    if float(opt_params.get("lambda_region", 0.0)) <= 0.0 and not bool(opt_params.get("region_loss_eval_metrics", False)):
        return None
    return RegionMaskProjector(
        gaussians.flame_verts,
        RegionMaskConfig(
            feather_px=float(opt_params.get("region_mask_feather_px", 3.0)),
            mouth_scale=float(opt_params.get("region_mask_mouth_scale", 1.18)),
            eyes_scale=float(opt_params.get("region_mask_eyes_scale", 1.20)),
            brow_scale=float(opt_params.get("region_mask_brow_scale", 1.18)),
            cheeks_scale=float(opt_params.get("region_mask_cheeks_scale", 1.08)),
            min_valid_pixels=float(opt_params.get("region_mask_min_valid_pixels", 8.0)),
        ),
    )


def build_region_masks(region_projector, gaussians, viewpoint_cam):
    if region_projector is None:
        return {}
    verts = getattr(gaussians, "current_flame_verts_for_region_masks", None)
    if verts is None:
        return {}
    return region_projector.build_masks(verts, viewpoint_cam)


def compute_region_loss_terms(image, gt_image, masks, opt_params):
    if not masks:
        return {}, {}
    loss_type = opt_params.get("region_loss_type", "charbonnier")
    weights = {
        "mouth": float(opt_params.get("region_w_mouth", 1.0)),
        "eyes": float(opt_params.get("region_w_eyes", 1.0)),
        "brow": float(opt_params.get("region_w_brow", 1.0)),
        "cheeks": float(opt_params.get("region_w_cheeks", 0.0)),
    }
    raw_losses = {
        name: masked_region_loss(image, gt_image, mask, loss_type=loss_type)
        for name, mask in masks.items()
    }
    weighted = {
        f"region_{name}": raw_losses[name] * weights[name]
        for name in raw_losses
    }
    metrics = region_l1_metrics(image.detach(), gt_image.detach(), masks)
    for name, value in raw_losses.items():
        metrics[f"region/{name}_{loss_type}_raw"] = float(value.detach().cpu())
    return weighted, metrics


def grad_norm_wrt_deform(loss, gaussians, retain_graph=True):
    deform = getattr(gaussians, "deform_output", None)
    if deform is None or not deform.requires_grad:
        return 0.0
    grad = torch.autograd.grad(
        loss,
        deform,
        retain_graph=retain_graph,
        allow_unused=True,
    )[0]
    if grad is None:
        return 0.0
    return float(grad.detach().norm().cpu())


def save_tensor_image(tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().float().cpu().clamp(0., 1.)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    elif image.ndim == 3:
        image = image.permute(1, 2, 0)
    array = (image.numpy() * 255.).round().astype("uint8")
    Image.fromarray(array).save(path)


def linear_warmup_scale(iteration: int, start_iter: int, warmup_iters: int, end_iter: int = None) -> float:
    if iteration < start_iter:
        return 0.0
    if end_iter is not None and end_iter >= 0 and iteration > end_iter:
        return 0.0
    if warmup_iters <= 0:
        return 1.0
    return max(0.0, min(1.0, float(iteration - start_iter + 1) / float(warmup_iters)))


def collect_cap4d_source_image_order(source_paths):
    image_paths = []
    for source_path in source_paths:
        source_path = Path(source_path)
        flame_paths = sorted((source_path / "flame").glob("*.npz"))
        image_paths_raw = sorted(
            [
                path for path in (source_path / "images").glob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS
            ]
        )
        images_by_stem = {}
        duplicates = []
        for path in image_paths_raw:
            if path.stem in images_by_stem:
                duplicates.append(path.stem)
            images_by_stem[path.stem] = path
        if duplicates:
            duplicate_preview = ", ".join(sorted(set(duplicates))[:20])
            raise ValueError(f"Duplicate image stems in {source_path / 'images'}: {duplicate_preview}")
        if len(flame_paths) == 0:
            raise FileNotFoundError(f"No FLAME files found in {source_path / 'flame'}")
        if len(images_by_stem) == 0:
            raise FileNotFoundError(f"No images found in {source_path / 'images'}")

        flame_stems = {path.stem for path in flame_paths}
        image_stems = set(images_by_stem.keys())
        missing_images = sorted(flame_stems - image_stems)
        missing_flames = sorted(image_stems - flame_stems)
        if missing_images or missing_flames:
            details = [
                f"Image/FLAME stem mismatch in {source_path}: "
                f"images={len(images_by_stem)}, flame={len(flame_paths)}"
            ]
            if missing_images:
                details.append("FLAME files without images: " + ", ".join(missing_images[:20]))
            if missing_flames:
                details.append("Images without FLAME files: " + ", ".join(missing_flames[:20]))
            raise ValueError("\n".join(details))
        image_paths.extend(images_by_stem[flame_path.stem] for flame_path in flame_paths)
    return image_paths


def _normalized_path_for_compare(path):
    return os.path.normcase(str(Path(path).resolve()))


def _frame_index_values_from_json(index_path):
    with index_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "image_paths" in data:
        return data["image_paths"]
    raise ValueError(
        f"Unsupported Xnemo frame index format in {index_path}: "
        "expected a list of image paths or a dict with image_paths."
    )


def _load_motion_feature_index(motion_feature_path):
    candidates = [
        Path(str(motion_feature_path) + ".frames.json"),
        motion_feature_path.with_suffix(".frames.json"),
    ]
    index_path = next((path for path in candidates if path.exists()), None)
    if index_path is not None:
        return index_path, _frame_index_values_from_json(index_path)

    shards_path = Path(str(motion_feature_path) + ".shards.json")
    if not shards_path.exists():
        return None, None

    with shards_path.open("r", encoding="utf-8") as f:
        shards_manifest = json.load(f)
    shard_inputs = shards_manifest.get("inputs", None)
    if not shard_inputs:
        raise ValueError(f"Xnemo shards manifest has no inputs: {shards_path}")

    shard_records = []
    for shard_input in shard_inputs:
        shard_feature_path = Path(shard_input)
        if not shard_feature_path.is_absolute():
            shard_feature_path = (shards_path.parent / shard_feature_path).resolve()
            if not shard_feature_path.exists():
                shard_feature_path = (Path.cwd() / shard_input).resolve()
        shard_index_path = Path(str(shard_feature_path) + ".frames.json")
        if not shard_index_path.exists():
            raise FileNotFoundError(
                f"Missing shard frame index for {shard_feature_path}: {shard_index_path}"
            )
        with shard_index_path.open("r", encoding="utf-8") as f:
            shard_index = json.load(f)
        if not isinstance(shard_index, dict) or "image_paths" not in shard_index:
            raise ValueError(
                f"Shard frame index must be a dict with image_paths: {shard_index_path}"
            )
        shard_records.append(
            {
                "path": shard_index_path,
                "shard_index": int(shard_index.get("shard_index", len(shard_records))),
                "shard_start": int(shard_index.get("shard_start", 0)),
                "shard_end": int(shard_index.get("shard_end", 0)),
                "image_paths": shard_index["image_paths"],
            }
        )

    shard_records.sort(key=lambda item: (item["shard_index"], item["shard_start"]))
    observed = []
    for record in shard_records:
        expected_len = max(0, record["shard_end"] - record["shard_start"])
        if expected_len and expected_len != len(record["image_paths"]):
            raise ValueError(
                f"Shard frame count mismatch in {record['path']}: "
                f"range={expected_len}, image_paths={len(record['image_paths'])}"
            )
        observed.extend(record["image_paths"])
    if "num_frames" in shards_manifest and int(shards_manifest["num_frames"]) != len(observed):
        raise ValueError(
            f"Xnemo shards manifest num_frames mismatch: "
            f"manifest={shards_manifest['num_frames']}, combined_index={len(observed)}"
        )
    return shards_path, observed


def validate_motion_feature_index(source_paths, motion_feature_path):
    if source_paths is None:
        return
    motion_feature_path = Path(motion_feature_path)
    index_path, observed_raw = _load_motion_feature_index(motion_feature_path)
    if index_path is None:
        print(
            "WARNING: No Xnemo frame index json found next to motion features. "
            "Expected .frames.json or .shards.json next to " + str(motion_feature_path)
        )
        return

    expected = [
        _normalized_path_for_compare(path)
        for path in collect_cap4d_source_image_order(source_paths)
    ]
    observed = [_normalized_path_for_compare(path) for path in observed_raw]

    if len(expected) != len(observed):
        raise ValueError(
            f"Xnemo frame index length mismatch: index={len(observed)}, "
            f"source_paths={len(expected)}. Re-extract motion features with the exact training source_paths."
        )
    for idx, (expected_path, observed_path) in enumerate(zip(expected, observed)):
        if expected_path != observed_path:
            raise ValueError(
                "Xnemo frame index order mismatch at frame "
                f"{idx}: expected {expected_path}, got {observed_path}. "
                "Re-extract motion features with the exact same source_paths and order."
            )
    print(f"Verified Xnemo frame index: {index_path} ({len(expected)} frames)")


def training(
    source_paths,
    model_path,
    model_params, 
    opt_params, 
    testing_iterations, 
    checkpoint_iterations, 
    load_existing_checkpoint, 
    init_checkpoint_path=None,
    enable_pseudo_back=False,
    pseudo_back_json=None,
    lambda_back_rgb=0.3,
    lambda_back_lpips=0.02,
    lambda_back_sil=0.2,
    back_start_iter=10000,
    back_end_iter=None,
    back_warmup_iters=10000,
    back_sample_ratio=0.10,
    pseudo_back_densify_start_iter=-1,
    lambda_motion_residual_l2=0.0,
    lambda_motion_residual_lap=0.0,
    lambda_motion_residual_ratio=0.0,
    motion_residual_ratio_limit=0.35,
    lambda_motion_mismatch_suppression=0.0,
    condition_monitor_interval=0,
    source_sampling_probabilities=None,
    source_regularizer_scales=None,
):
    first_iter = 0
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(model_path)
    else:
        print("Tensorboard not available: not logging progress")
    
    gaussians = CAP4DGaussianModel(model_params)
    
    scene = Scene(
        model_path=model_path, 
        source_paths=source_paths, 
        gaussians=gaussians,
        enable_pseudo_back=enable_pseudo_back,
        pseudo_back_paths=pseudo_back_json,
    )
    if init_checkpoint_path is not None:
        init_checkpoint_path = Path(init_checkpoint_path)
        if not init_checkpoint_path.exists():
            raise FileNotFoundError(f"Initialization checkpoint does not exist: {init_checkpoint_path}")
        if load_existing_checkpoint:
            raise ValueError("Use either init_checkpoint_path or load_existing_checkpoint, not both.")
        model_weights, init_iteration = torch.load(init_checkpoint_path, weights_only=False)
        gaussians.restore(model_weights, training_args=None)
        print(
            "Initialized model weights without optimizer state:",
            f"checkpoint={init_checkpoint_path}",
            f"source_iteration={init_iteration}",
        )
    region_projector = make_region_projector(gaussians, opt_params)
    scene.region_projector = region_projector
    if region_projector is not None:
        print(
            "Enabled region mask projector:",
            f"lambda_region={float(opt_params.get('lambda_region', 0.0))}",
            f"loss_type={opt_params.get('region_loss_type', 'charbonnier')}",
            f"vertex_counts={region_projector.region_vertex_counts}",
        )
    gaussians.training_setup(opt_params)

    # if prompted and if it exists, load existing checkpoint
    if load_existing_checkpoint:
        loaded_iter, chkpt_path = searchForMaxIteration(model_path)
        if loaded_iter is None:
            print("WARNING: No valid checkpoint found in ", model_path)
        else:
            print("Loading trained model at iteration {}".format(loaded_iter))
            (model_weights, first_iter) = torch.load(chkpt_path, weights_only=False)
            gaussians.restore(model_weights, opt_params)
    
    lpips = LPIPS('vgg').cuda()

    bg_color = [1, 1, 1]  # force white background
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    loader_camera_train = None
    iter_camera_train = None
    loader_camera_real = None
    iter_camera_real = None
    loader_camera_pseudo = None
    iter_camera_pseudo = None
    realized_source_probabilities = None
    real_source_counts = None

    if enable_pseudo_back:
        train_cameras = scene.train_cameras[1.0]
        real_train_cameras = [camera for camera in train_cameras if not getattr(camera, "is_pseudo", False)]
        pseudo_train_cameras = [camera for camera in train_cameras if getattr(camera, "is_pseudo", False)]
        print(f"Training cameras: real={len(real_train_cameras)}, pseudo_back={len(pseudo_train_cameras)}")
        if len(pseudo_train_cameras) == 0:
            print("WARNING: enable_pseudo_back=True but no pseudo back cameras are available")
        if back_end_iter is None or back_end_iter < 0:
            back_end_iter = opt_params["iterations"]
        back_sample_ratio = max(0.0, min(1.0, back_sample_ratio))
        print(
            "Pseudo back schedule:",
            f"enable={enable_pseudo_back}",
            f"start={back_start_iter}",
            f"end={back_end_iter}",
            f"warmup={back_warmup_iters}",
            f"max_sample_ratio={back_sample_ratio}",
            f"lambda_rgb={lambda_back_rgb}",
            f"lambda_lpips={lambda_back_lpips}",
            f"lambda_sil={lambda_back_sil}",
            f"densify_start={pseudo_back_densify_start_iter}",
        )

        loader_camera_real, realized_source_probabilities, real_source_counts = make_camera_loader(
            real_train_cameras,
            source_sampling_probabilities,
            seed=opt_params.get("seed", 0),
            num_workers=8,
        )
        iter_camera_real = iter(loader_camera_real)

        if len(pseudo_train_cameras) > 0:
            pseudo_num_workers = min(2, len(pseudo_train_cameras))
            loader_camera_pseudo = DataLoader(
                CameraDataset(pseudo_train_cameras),
                batch_size=None,
                shuffle=True,
                num_workers=pseudo_num_workers,
                pin_memory=True,
                persistent_workers=pseudo_num_workers > 0
            )
            iter_camera_pseudo = iter(loader_camera_pseudo)
    else:
        train_cameras = scene.train_cameras[1.0]
        loader_camera_train, realized_source_probabilities, real_source_counts = make_camera_loader(
            train_cameras,
            source_sampling_probabilities,
            seed=opt_params.get("seed", 0),
            num_workers=8,
        )
        iter_camera_train = iter(loader_camera_train)

    if realized_source_probabilities is not None:
        print(
            "Source-balanced sampling:",
            f"probabilities={realized_source_probabilities.tolist()}",
            f"train_counts={real_source_counts.tolist()}",
        )
        for idx, source_path in enumerate(source_paths):
            print(
                f"  source[{idx}]",
                f"path={source_path}",
                f"train_count={int(real_source_counts[idx])}",
                f"target_probability={float(realized_source_probabilities[idx]):.6f}",
            )
    if source_regularizer_scales is not None:
        print(
            "Per-source deformation regularizer scales:",
            source_regularizer_scales,
            "(applies to laplacian, relative_deform, motion_residual_l2, motion_residual_lap)",
        )

    ema_loss_for_log = 0.0
    pseudo_sample_count = 0
    source_sample_counts = [0 for _ in (source_paths or [])]
    progress_bar = tqdm(range(first_iter, opt_params["iterations"]), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt_params["iterations"] + 1):    
        iter_start.record()

        gaussians.train()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if (
            not getattr(gaussians, "motion_cross_attention_adapter_only", False)
            and iteration % opt_params["sh_warmup_iterations"] == 0
        ):
            gaussians.oneupSHdegree()

        if enable_pseudo_back:
            use_pseudo_back = False
            back_schedule_scale = linear_warmup_scale(
                iteration,
                back_start_iter,
                back_warmup_iters,
                back_end_iter,
            )
            current_back_sample_ratio = back_sample_ratio * back_schedule_scale
            if loader_camera_pseudo is not None and current_back_sample_ratio > 0.0:
                use_pseudo_back = torch.rand((), device="cpu").item() < current_back_sample_ratio

            if use_pseudo_back:
                viewpoint_cam, iter_camera_pseudo = next_camera_from_loader(loader_camera_pseudo, iter_camera_pseudo)
                pseudo_sample_count += 1
            else:
                viewpoint_cam, iter_camera_real = next_camera_from_loader(loader_camera_real, iter_camera_real)
        else:
            viewpoint_cam, iter_camera_train = next_camera_from_loader(loader_camera_train, iter_camera_train)

        is_pseudo_view = getattr(viewpoint_cam, "is_pseudo", False)
        source_id = getattr(viewpoint_cam, "source_id", None)
        if not is_pseudo_view and source_id is not None:
            source_sample_counts[int(source_id)] += 1
        deformation_regularizer_scale = source_value_for_camera(
            viewpoint_cam,
            source_regularizer_scales,
            default=1.0,
        )

        # Set timestep and run FLAME model
        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(viewpoint_cam.timestep)

        # Render Gaussians
        render_pkg = render(
            viewpoint_cam, 
            gaussians, 
            background, 
        )
        image = render_pkg["render"]
        alpha = render_pkg["alpha"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # load gt image and mask
        gt_image = viewpoint_cam.original_image.cuda()
        mask = viewpoint_cam.mask[..., None].cuda().float().permute(2, 0, 1)
        assert mask.shape[1] == image.shape[1] and mask.shape[2] == image.shape[2]

        # Loss computation
        losses = {}
        region_logs = {}
        motion_residual_ratio_value = None

        lambda_lpips = 0.
        if is_pseudo_view:
            pseudo_weight = float(getattr(viewpoint_cam, "pseudo_weight", 1.0))
            pseudo_weight *= linear_warmup_scale(
                iteration,
                back_start_iter,
                back_warmup_iters,
                back_end_iter,
            )

            masked_image = image * mask
            masked_gt_image = gt_image * mask
            losses['back_rgb'] = masked_l1_loss(image, gt_image, mask) * lambda_back_rgb * pseudo_weight
            losses['back_lpips'] = lpips(masked_image[None], masked_gt_image[None]).mean() * lambda_back_lpips * pseudo_weight
            losses['back_sil'] = l1_loss(alpha, mask) * lambda_back_sil * pseudo_weight
            losses['l1'] = torch.tensor(0., device="cuda")
            losses['ssim'] = torch.tensor(0., device="cuda")
            losses['lpips'] = torch.tensor(0., device="cuda")
        else:
            image = image * mask
            gt_image = gt_image * mask

            if iteration > opt_params["lpips_linear_start"]:
                lambda_lpips = (iteration - opt_params["lpips_linear_start"]) / (opt_params["lpips_linear_end"] - opt_params["lpips_linear_start"]) * opt_params["lambda_lpips_end"]
                lambda_lpips = min(lambda_lpips, opt_params["lambda_lpips_end"])
                losses['lpips'] = opt_params["w_lpips"] * lambda_lpips * lpips(image, gt_image)
            else:
                losses['lpips'] = torch.tensor(0., device="cuda")

            losses['l1'] = l1_loss(image, gt_image) * (1.0 - opt_params["lambda_dssim"]) * (1.0 - lambda_lpips)
            losses['ssim'] = (1.0 - ssim(image, gt_image)) * opt_params["lambda_dssim"] * (1.0 - lambda_lpips)
            losses['real_rgb'] = losses['l1'] + losses['ssim']

            lambda_region = float(opt_params.get("lambda_region", 0.0))
            if region_projector is not None:
                region_masks = build_region_masks(region_projector, gaussians, viewpoint_cam)
                region_terms, region_logs = compute_region_loss_terms(image, gt_image, region_masks, opt_params)
                if lambda_region > 0.0 and region_terms:
                    region_weighted_terms = {
                        name: value * lambda_region
                        for name, value in region_terms.items()
                    }
                    losses.update(region_weighted_terms)
                    losses["region_total"] = sum(region_weighted_terms.values())

        if opt_params["metric_xyz"]:
            losses['xyz'] = F.relu((gaussians._xyz*gaussians.face_scaling[gaussians.binding])[visibility_filter] - opt_params["threshold_xyz"]).norm(dim=1).mean() * opt_params["lambda_xyz"]
        else:
            losses['xyz'] = F.relu(gaussians._xyz[visibility_filter].norm(dim=1) - opt_params["threshold_xyz"]).mean() * opt_params["lambda_xyz"]

        if opt_params["lambda_scale"] != 0:
            if opt_params["metric_scale"]:
                losses['scale'] = F.relu(gaussians.get_scaling[visibility_filter] - opt_params["threshold_scale"]).norm(dim=1).mean() * opt_params["lambda_scale"]
            else:
                losses['scale'] = F.relu(torch.exp(gaussians._scaling[visibility_filter]) - opt_params["threshold_scale"]).norm(dim=1).mean() * opt_params["lambda_scale"]

        if opt_params["lambda_laplacian"] != 0:
            losses['lap'] = (
                gaussians.compute_laplacian_loss()
                * opt_params["lambda_laplacian"]
                * deformation_regularizer_scale
            )

        if lambda_motion_residual_l2 != 0:
            losses['motion_res_l2'] = (
                gaussians.compute_motion_residual_l2_loss()
                * lambda_motion_residual_l2
                * deformation_regularizer_scale
            )

        if lambda_motion_residual_lap != 0:
            losses['motion_res_lap'] = (
                gaussians.compute_motion_residual_laplacian_loss()
                * lambda_motion_residual_lap
                * deformation_regularizer_scale
            )

        if lambda_motion_residual_ratio != 0:
            ratio_penalty, residual_ratio = gaussians.compute_motion_residual_ratio_loss(
                motion_residual_ratio_limit
            )
            losses['motion_res_ratio'] = (
                ratio_penalty
                * lambda_motion_residual_ratio
                * deformation_regularizer_scale
            )
            motion_residual_ratio_value = float(residual_ratio.detach().cpu())

        if lambda_motion_mismatch_suppression != 0:
            losses['motion_mismatch'] = (
                gaussians.compute_motion_mismatch_suppression_loss()
                * lambda_motion_mismatch_suppression
            )

        if opt_params["lambda_relative_deform"] != 0:
            losses['deform'] = (
                gaussians.compute_relative_deformation_loss()
                * opt_params["lambda_relative_deform"]
                * deformation_regularizer_scale
            )

        if opt_params["lambda_relative_rot"] != 0:
            losses['rot'] = gaussians.compute_relative_rotation_loss() * opt_params["lambda_relative_rot"]

        if opt_params["lambda_neck"] != 0:
            losses['neck'] = gaussians.compute_neck_loss() * opt_params["lambda_neck"]
        
        calibration_interval = int(opt_params.get("region_calibration_interval", 0))
        if calibration_interval > 0 and iteration % calibration_interval == 0 and not is_pseudo_view:
            global_recon_loss = (
                losses.get("l1", torch.tensor(0., device="cuda"))
                + losses.get("ssim", torch.tensor(0., device="cuda"))
                + losses.get("lpips", torch.tensor(0., device="cuda"))
            )
            region_loss_unscaled = sum(
                value / max(float(opt_params.get("lambda_region", 0.0)), 1e-12)
                for key, value in losses.items()
                if key.startswith("region_") and key != "region_total"
            )
            if isinstance(region_loss_unscaled, torch.Tensor) and region_loss_unscaled.requires_grad:
                region_logs["region/calibration/global_grad_norm_deform"] = grad_norm_wrt_deform(
                    global_recon_loss, gaussians, retain_graph=True
                )
                region_logs["region/calibration/region_grad_norm_deform_unscaled"] = grad_norm_wrt_deform(
                    region_loss_unscaled, gaussians, retain_graph=True
                )
                g = region_logs["region/calibration/global_grad_norm_deform"]
                r = region_logs["region/calibration/region_grad_norm_deform_unscaled"]
                target_ratio = float(opt_params.get("region_calibration_target_ratio", 0.3))
                if r > 0.0:
                    region_logs["region/calibration/recommended_lambda_region"] = target_ratio * g / r

        log_only_losses = {"real_rgb", "region_total"}
        losses['total'] = sum([v for k, v in losses.items() if k not in log_only_losses])
        losses['total'].backward()

        condition_monitor_logs = {}
        if condition_monitor_interval > 0 and iteration % condition_monitor_interval == 0:
            condition_monitor_logs = gaussians.collect_condition_monitor_stats(include_sensitivity=True)
            with torch.no_grad():
                condition_monitor_logs.update(
                    compute_render_condition_sensitivity(
                        scene,
                        render,
                        background,
                        viewpoint_cam,
                    )
                )
            if condition_monitor_logs:
                monitor_preview = {
                    key: condition_monitor_logs[key]
                    for key in (
                        "condition/site_count",
                        "condition/shared_trunk_grad_norm",
                        "condition/spatial_residual_v2/delta_actual_mean_abs",
                        "condition/spatial_residual_v2/delta_over_base",
                        "condition/spatial_residual_v2/sensitivity/fixed_uv_alt_condition_delta_mean_abs",
                        "condition/spatial_residual_v2/sensitivity/delta_ratio_cond_over_uv",
                        "condition/cross_attention/site_count",
                        "condition/cross_attention/param_count",
                        "condition/cross_attention/input_uv_dropout_fraction",
                        "condition/cross_attention/input_uv_noise_std",
                        "condition/cross_attention/base_pretrain_active",
                        "condition/cross_attention/base_lr_scale",
                        "condition/cross_attention/condition_lr_scale",
                        "condition/cross_attention/adapter_only",
                        "condition/cross_attention_v3/delta_actual_mean_abs",
                        "condition/cross_attention_v3/delta_actual_rms",
                        "condition/cross_attention_v3/base_actual_rms",
                        "condition/cross_attention_v3/delta_over_base_rms",
                        "condition/cross_attention_v3/nodeform_delta_actual_max_abs",
                        "condition/cross_attention_v4/feature_common_energy_ratio",
                        "condition/cross_attention_v4/current_raw_rms",
                        "condition/cross_attention_v4/current_centered_rms",
                        "condition/cross_attention_v4/mismatch_index",
                        "condition/cross_attention_v4/mismatch_flame_distance",
                        "condition/cross_attention_v4/mismatch_condition_cosine",
                        "condition/cross_attention_v4/delta_actual_rms",
                        "condition/cross_attention_v4/base_actual_rms",
                        "condition/cross_attention_v4/delta_over_base_rms",
                        "condition/cross_attention_v4/mismatch_delta_actual_rms",
                        "condition/cross_attention_v4/mismatch_over_base_rms",
                        "condition/cross_attention_v4/aligned_over_mismatch_rms",
                        "condition/cross_attention_v4/nodeform_delta_actual_max_abs",
                        "condition/cross_attention/scale_1x/skip_cross_attention_c128/gate",
                        "condition/cross_attention/scale_1x/skip_cross_attention_c128/logit_scale",
                        "condition/cross_attention/scale_1x/skip_cross_attention_c128/attention_entropy_mean",
                        "condition/cross_attention/scale_1x/skip_cross_attention_c128/attention_max_mean",
                        "condition/cross_attention/scale_1x/skip_cross_attention_c128/token_grad_norm",
                        "condition/cross_attention/scale_1x/skip_cross_attention_c128/query_grad_norm",
                        "condition/cross_attention/scale_1x/skip_cross_attention_c128/key_grad_norm",
                        "condition/cross_attention/scale_1x/skip_cross_attention_c128/value_grad_norm",
                        "condition/cross_attention/scale_1x/skip_cross_attention_c128/out_grad_norm",
                        "condition/cross_attention/scale_1x/decoder_cross_attention_c64/gate",
                        "condition/cross_attention/scale_1x/decoder_cross_attention_c64/attention_entropy_mean",
                        "condition/cross_attention/scale_1x/decoder_cross_attention_c64/token_grad_norm",
                        "condition/cross_attention/outermost/output_cross_attention_c3/gate",
                        "condition/cross_attention/outermost/output_cross_attention_c3/attention_entropy_mean",
                        "condition/cross_attention/outermost/output_cross_attention_c3/token_grad_norm",
                        "condition/sensitivity/fixed_uv_alt_condition_mean_abs",
                        "condition/sensitivity/fixed_condition_alt_uv_mean_abs",
                        "condition/render_sensitivity/render_ratio_cond_over_uv",
                    )
                    if key in condition_monitor_logs
                }
                print(f"[ITER {iteration}] Condition monitor: {json.dumps(monitor_preview, sort_keys=True)}")
        if region_logs and int(opt_params.get("region_log_interval", 100)) > 0 and iteration % int(opt_params.get("region_log_interval", 100)) == 0:
            preview_keys = [
                "region/full_face_l1",
                "region/mouth_l1",
                "region/eyes_l1",
                "region/brow_l1",
                "region/cheeks_l1",
                "region/calibration/global_grad_norm_deform",
                "region/calibration/region_grad_norm_deform_unscaled",
                "region/calibration/recommended_lambda_region",
            ]
            preview = {key: region_logs[key] for key in preview_keys if key in region_logs}
            if preview:
                print(f"[ITER {iteration}] Region monitor: {json.dumps(preview, sort_keys=True)}")

        real_sample_total = sum(source_sample_counts)
        source_logs = {
            "source_sampling/current_source_id": float(source_id) if source_id is not None else -1.0,
            "regularizer/deformation_source_scale": deformation_regularizer_scale,
            "regularizer/effective_lambda_laplacian": (
                float(opt_params["lambda_laplacian"]) * deformation_regularizer_scale
            ),
            "regularizer/effective_lambda_relative_deform": (
                float(opt_params["lambda_relative_deform"]) * deformation_regularizer_scale
            ),
            "regularizer/motion_residual_ratio": (
                motion_residual_ratio_value
                if motion_residual_ratio_value is not None else 0.0
            ),
            "regularizer/motion_residual_ratio_limit": float(motion_residual_ratio_limit),
            "regularizer/effective_lambda_motion_residual_ratio": (
                float(lambda_motion_residual_ratio) * deformation_regularizer_scale
            ),
            "regularizer/lambda_motion_mismatch_suppression": float(
                lambda_motion_mismatch_suppression
            ),
        }
        for idx, count in enumerate(source_sample_counts):
            source_logs[f"source_sampling/source_{idx}_count"] = count
            source_logs[f"source_sampling/source_{idx}_fraction"] = (
                count / real_sample_total if real_sample_total > 0 else 0.0
            )
            if realized_source_probabilities is not None:
                source_logs[f"source_sampling/source_{idx}_target"] = float(
                    realized_source_probabilities[idx]
                )
        source_log_interval = int(opt_params.get("region_log_interval", 100))
        if (
            realized_source_probabilities is not None
            and source_log_interval > 0
            and iteration % source_log_interval == 0
        ):
            source_preview = {
                key: value
                for key, value in source_logs.items()
                if key.endswith("_fraction")
                or key.endswith("_target")
                or key.startswith("regularizer/")
            }
            print(f"[ITER {iteration}] Source monitor: {json.dumps(source_preview, sort_keys=True)}")

        iter_end.record()

        with torch.no_grad():
            gaussians.eval()

            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if 'xyz' in losses:
                    postfix["xyz"] = f"{losses['xyz']:.{7}f}"
                if 'scale' in losses:
                    postfix["scale"] = f"{losses['scale']:.{7}f}"
                if 'dy_off' in losses:
                    postfix["dy_off"] = f"{losses['dy_off']:.{7}f}"
                if 'lap' in losses:
                    postfix["lap"] = f"{losses['lap']:.{7}f}"
                if 'motion_res_l2' in losses:
                    postfix["motion_res_l2"] = f"{losses['motion_res_l2']:.{7}f}"
                if 'motion_res_lap' in losses:
                    postfix["motion_res_lap"] = f"{losses['motion_res_lap']:.{7}f}"
                if 'motion_mismatch' in losses:
                    postfix["motion_mismatch"] = f"{losses['motion_mismatch']:.{7}f}"
                if 'region_total' in losses:
                    postfix["region"] = f"{losses['region_total']:.{7}f}"
                if 'back_rgb' in losses:
                    postfix["back_rgb"] = f"{losses['back_rgb']:.{7}f}"
                if 'back_sil' in losses:
                    postfix["back_sil"] = f"{losses['back_sil']:.{7}f}"
                if 'dynamic_offset_std' in losses:
                    postfix["dynamic_offset_std"] = f"{losses['dynamic_offset_std']:.{7}f}"
                postfix["pseudo"] = pseudo_sample_count
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt_params["iterations"]:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer, 
                iteration, 
                losses, 
                {
                    "pseudo_sample_count": pseudo_sample_count,
                    **source_logs,
                    **condition_monitor_logs,
                    **region_logs,
                },
                iter_start.elapsed_time(iter_end), 
                testing_iterations, 
                scene, 
                render, 
                background,
                lpips,
            )

            if enable_pseudo_back and iteration in testing_iterations:
                save_pseudo_back_visualizations(
                    scene,
                    render,
                    background,
                    iteration,
                    Path(model_path),
                    tb_writer,
                )

            # Densification
            if (
                not getattr(gaussians, "motion_cross_attention_adapter_only", False)
                and iteration < opt_params["densify_until_iter"]
            ):
                # Keep track of max radii in image-space for pruning
                allow_pseudo_densify = (
                    not is_pseudo_view
                    or (
                        pseudo_back_densify_start_iter >= 0
                        and iteration >= pseudo_back_densify_start_iter
                    )
                )
                if allow_pseudo_densify:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt_params["densify_from_iter"] and iteration % opt_params["densification_interval"] == 0:
                    size_threshold = 20 if iteration > opt_params["opacity_reset_interval"] else None
                    gaussians.densify_and_prune(opt_params["densify_grad_threshold"], 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt_params["opacity_reset_interval"] == 0 or (iteration == opt_params["densify_from_iter"]):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt_params["iterations"]:
                gaussians.optimizer_step()

            if (iteration in checkpoint_iterations):
                print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def save_pseudo_back_visualizations(
    scene: Scene,
    renderFunc,
    background,
    iteration,
    model_path: Path,
    tb_writer=None,
):
    pseudo_cameras = [
        camera for camera in scene.train_cameras[1.0]
        if getattr(camera, "is_pseudo", False) and getattr(camera, "is_back_view", False)
    ]
    if len(pseudo_cameras) == 0:
        return

    output_dir = model_path / "pseudo_back_vis" / f"iter_{iteration:06d}"
    scene.gaussians.eval()
    for camera in pseudo_cameras:
        viewpoint = CameraDataset([camera])[0]
        if scene.gaussians.binding != None:
            scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)

        render_pkg = renderFunc(viewpoint, scene.gaussians, background)
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        alpha = torch.clamp(render_pkg["alpha"], 0.0, 1.0)
        mask = viewpoint.mask.cuda().float()[None]
        name = str(getattr(viewpoint, "image_name", f"pseudo_{viewpoint.uid}"))

        save_tensor_image(image, output_dir / f"{name}_render.png")
        save_tensor_image(alpha, output_dir / f"{name}_alpha.png")
        save_tensor_image(mask, output_dir / f"{name}_mask.png")

        if tb_writer:
            tb_writer.add_images(f"pseudo_back/{name}_render", image[None], global_step=iteration)
            tb_writer.add_images(f"pseudo_back/{name}_alpha", alpha[None], global_step=iteration)
            tb_writer.add_images(f"pseudo_back/{name}_mask", mask[None], global_step=iteration)


def compute_render_condition_sensitivity(scene, renderFunc, background, viewpoint_cam):
    gaussians = scene.gaussians
    if (
        not getattr(gaussians, "use_conditional_norm", False)
        or getattr(gaussians, "motion_features", None) is None
        or getattr(gaussians, "num_timesteps", 0) <= 1
    ):
        return {}

    timestep = int(viewpoint_cam.timestep)
    n_motion = int(gaussians.motion_features.shape[0])
    alt_timestep = (timestep + max(1, n_motion // 2)) % n_motion
    current_feature = gaussians.motion_features[[timestep]].detach()
    alt_feature = gaussians.motion_features[[alt_timestep]].detach()

    was_training = getattr(gaussians, "_is_training_mode", False)
    gaussians.eval()
    try:
        gaussians.clear_motion_feature_override()
        gaussians.select_mesh_by_timestep(timestep)
        render_real = renderFunc(viewpoint_cam, gaussians, background)["render"].detach()
        deform_real = gaussians.deform_output.detach()
        delta_real = (
            gaussians.spatial_residual_v2_deform_output.detach()
            if getattr(gaussians, "spatial_residual_v2_deform_output", None) is not None
            else None
        )
        region_masks = build_region_masks(getattr(scene, "region_projector", None), gaussians, viewpoint_cam)

        gaussians.set_motion_feature_override(alt_feature)
        gaussians.select_mesh_by_timestep(timestep)
        render_cond_swap = renderFunc(viewpoint_cam, gaussians, background)["render"].detach()
        deform_cond_swap = gaussians.deform_output.detach()
        delta_cond_swap = (
            gaussians.spatial_residual_v2_deform_output.detach()
            if getattr(gaussians, "spatial_residual_v2_deform_output", None) is not None
            else None
        )

        gaussians.set_motion_feature_override(current_feature)
        gaussians.select_mesh_by_timestep(alt_timestep)
        render_uv_swap = renderFunc(viewpoint_cam, gaussians, background)["render"].detach()
        deform_uv_swap = gaussians.deform_output.detach()
        delta_uv_swap = (
            gaussians.spatial_residual_v2_deform_output.detach()
            if getattr(gaussians, "spatial_residual_v2_deform_output", None) is not None
            else None
        )

        cond_deform_diff = (deform_real - deform_cond_swap).abs()
        uv_deform_diff = (deform_real - deform_uv_swap).abs()
        cond_render_diff = (render_real - render_cond_swap).abs()
        uv_render_diff = (render_real - render_uv_swap).abs()
        eps = 1e-8
        uv_region_masks = {}
        region_projector = getattr(scene, "region_projector", None)
        if (
            region_projector is not None
            and delta_real is not None
            and delta_cond_swap is not None
            and delta_uv_swap is not None
        ):
            uv_region_masks = region_projector.build_uv_masks(
                gaussians.flame_faces,
                gaussians.fragments.pix_to_face,
                getattr(gaussians, "uv_mask", None),
                target_size=tuple(delta_real.shape[-2:]),
            )
        stats = {
            "condition/render_sensitivity/fixed_uv_alt_condition_mean_abs": float(cond_render_diff.mean().cpu()),
            "condition/render_sensitivity/fixed_uv_alt_condition_max_abs": float(cond_render_diff.max().cpu()),
            "condition/render_sensitivity/fixed_condition_alt_uv_mean_abs": float(uv_render_diff.mean().cpu()),
            "condition/render_sensitivity/fixed_condition_alt_uv_max_abs": float(uv_render_diff.max().cpu()),
            "condition/render_sensitivity/render_ratio_cond_over_uv": float(
                (cond_render_diff.mean() / (uv_render_diff.mean() + eps)).cpu()
            ),
            "condition/render_sensitivity/deform_ratio_cond_over_uv": float(
                (cond_deform_diff.mean() / (uv_deform_diff.mean() + eps)).cpu()
            ),
        }
        if delta_real is not None and delta_cond_swap is not None and delta_uv_swap is not None:
            cond_delta_diff = (delta_real - delta_cond_swap).abs()
            uv_delta_diff = (delta_real - delta_uv_swap).abs()
            stats.update(
                {
                    "condition/spatial_residual_v2/render_sensitivity/fixed_uv_alt_condition_delta_mean_abs": float(cond_delta_diff.mean().cpu()),
                    "condition/spatial_residual_v2/render_sensitivity/fixed_condition_alt_uv_delta_mean_abs": float(uv_delta_diff.mean().cpu()),
                    "condition/spatial_residual_v2/render_sensitivity/delta_ratio_cond_over_uv": float(
                        (cond_delta_diff.mean() / (uv_delta_diff.mean() + eps)).cpu()
                    ),
                }
            )
        for region_name, region_mask in region_masks.items():
            denom = region_mask.sum().clamp_min(1.0) * cond_render_diff.shape[0]
            cond_region = (cond_render_diff * region_mask).sum() / denom
            uv_region = (uv_render_diff * region_mask).sum() / denom
            stats[f"condition/render_sensitivity/{region_name}_fixed_uv_alt_condition_mean_abs"] = float(cond_region.cpu())
            stats[f"condition/render_sensitivity/{region_name}_fixed_condition_alt_uv_mean_abs"] = float(uv_region.cpu())
            stats[f"condition/render_sensitivity/{region_name}_ratio_cond_over_uv"] = float(
                (cond_region / (uv_region + eps)).cpu()
            )
        if delta_real is not None and delta_cond_swap is not None and delta_uv_swap is not None:
            for region_name, uv_region_mask in uv_region_masks.items():
                delta_denom = uv_region_mask.sum().clamp_min(1.0) * delta_real.shape[1]
                delta_region = (delta_real.abs() * uv_region_mask).sum() / delta_denom
                cond_delta_region = (cond_delta_diff * uv_region_mask).sum() / delta_denom
                uv_delta_region = (uv_delta_diff * uv_region_mask).sum() / delta_denom
                stats[f"condition/spatial_residual_v2/{region_name}_delta_actual_mean_abs"] = float(delta_region.cpu())
                stats[f"condition/spatial_residual_v2/render_sensitivity/{region_name}_fixed_uv_alt_condition_delta_mean_abs"] = float(cond_delta_region.cpu())
                stats[f"condition/spatial_residual_v2/render_sensitivity/{region_name}_fixed_condition_alt_uv_delta_mean_abs"] = float(uv_delta_region.cpu())
                stats[f"condition/spatial_residual_v2/render_sensitivity/{region_name}_delta_ratio_cond_over_uv"] = float(
                    (cond_delta_region / (uv_delta_region + eps)).cpu()
                )
    finally:
        gaussians.clear_motion_feature_override()
        gaussians.select_mesh_by_timestep(timestep)
        if was_training:
            gaussians.train()
        else:
            gaussians.eval()
    return stats


def training_report(
    tb_writer, 
    iteration, 
    losses, 
    extra_logs,
    elapsed, 
    testing_iterations, 
    scene: Scene, 
    renderFunc, 
    background,
    lpips: LPIPS,
):
    if tb_writer and iteration % 10 == 0:
        if 'l1' in losses:
            tb_writer.add_scalar('train_loss_patches/l1_loss', losses['l1'].detach().item(), iteration)
        if 'ssim' in losses:
            tb_writer.add_scalar('train_loss_patches/ssim_loss', losses['ssim'].detach().item(), iteration)
        if 'lpips' in losses:
            tb_writer.add_scalar('train_loss_patches/lpips', losses['lpips'].detach().item(), iteration)
        if 'real_rgb' in losses:
            tb_writer.add_scalar('train_loss_patches/loss_real_rgb', losses['real_rgb'].detach().item(), iteration)
        if 'back_rgb' in losses:
            tb_writer.add_scalar('train_loss_patches/loss_back_rgb', losses['back_rgb'].detach().item(), iteration)
        if 'back_lpips' in losses:
            tb_writer.add_scalar('train_loss_patches/loss_back_lpips', losses['back_lpips'].detach().item(), iteration)
        if 'back_sil' in losses:
            tb_writer.add_scalar('train_loss_patches/loss_back_sil', losses['back_sil'].detach().item(), iteration)
        if 'xyz' in losses:
            tb_writer.add_scalar('train_loss_patches/xyz_loss', losses['xyz'].detach().item(), iteration)
        if 'scale' in losses:
            tb_writer.add_scalar('train_loss_patches/scale_loss', losses['scale'].detach().item(), iteration)
        if 'dynamic_offset' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset', losses['dynamic_offset'].detach().item(), iteration)
        if 'lap' in losses:
            tb_writer.add_scalar('train_loss_patches/lap', losses['lap'].detach().item(), iteration)
        if 'motion_res_l2' in losses:
            tb_writer.add_scalar('train_loss_patches/motion_res_l2', losses['motion_res_l2'].detach().item(), iteration)
        if 'motion_res_lap' in losses:
            tb_writer.add_scalar('train_loss_patches/motion_res_lap', losses['motion_res_lap'].detach().item(), iteration)
        for key in ("region_mouth", "region_eyes", "region_brow", "region_cheeks", "region_total"):
            if key in losses:
                tb_writer.add_scalar(f"train_loss_patches/{key}", losses[key].detach().item(), iteration)
        if 'deform' in losses:
            tb_writer.add_scalar('train_loss_patches/deform', losses['deform'].detach().item(), iteration)
        if 'rot' in losses:
            tb_writer.add_scalar('train_loss_patches/rot', losses['rot'].detach().item(), iteration)
        if 'dynamic_offset_std' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset_std', losses['dynamic_offset_std'].detach().item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', losses['total'].detach().item(), iteration)
        tb_writer.add_scalar('train/pseudo_sample_count', extra_logs.get("pseudo_sample_count", 0), iteration)
        for key, value in extra_logs.items():
            if key.startswith(("condition/", "region/", "source_sampling/", "regularizer/")):
                tb_writer.add_scalar(key, float(value), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        print("[ITER {}] Evaluating".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'val', 'cameras' : scene.getValCameras()},
            {'name': 'test', 'cameras' : scene.getTestCameras()},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                region_metrics_enabled = getattr(scene, "region_projector", None) is not None
                region_metric_sums = {
                    "full_face_l1": 0.0,
                    "mouth_l1": 0.0,
                    "eyes_l1": 0.0,
                    "brow_l1": 0.0,
                    "cheeks_l1": 0.0,
                } if region_metrics_enabled else {}
                num_vis_img = 10
                image_cache = []
                gt_image_cache = []
                vis_ct = 0
                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    if scene.gaussians.num_timesteps > 1:
                        scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, background)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx % (len(config['cameras'])) // num_vis_img) == 0:
                        tb_writer.add_images(config['name'] + "_{}/render".format(vis_ct), image[None], global_step=iteration)
                        error_image = error_map(image, gt_image)
                        tb_writer.add_images(config['name'] + "_{}/error".format(vis_ct), error_image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(vis_ct), gt_image[None], global_step=iteration)
                        
                        # Visualize U-Net expression dependent deformation (manually normalized)
                        deform = scene.gaussians.deform_output / 0.0108 / 2. + 0.5
                        tb_writer.add_images(config['name'] + f"_{vis_ct}/deform", deform, global_step=iteration)

                        vis_ct += 1
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    if region_metrics_enabled:
                        masks = build_region_masks(scene.region_projector, scene.gaussians, viewpoint)
                        region_metrics = region_l1_metrics(image, gt_image, masks)
                        region_metric_sums["full_face_l1"] += region_metrics.get("region/full_face_l1", 0.0)
                        region_metric_sums["mouth_l1"] += region_metrics.get("region/mouth_l1", 0.0)
                        region_metric_sums["eyes_l1"] += region_metrics.get("region/eyes_l1", 0.0)
                        region_metric_sums["brow_l1"] += region_metrics.get("region/brow_l1", 0.0)
                        region_metric_sums["cheeks_l1"] += region_metrics.get("region/cheeks_l1", 0.0)

                    image_cache.append(image)
                    gt_image_cache.append(gt_image)

                    if idx == len(config['cameras']) - 1 or len(image_cache) == 16:
                        batch_img = torch.stack(image_cache, dim=0)
                        batch_gt_img = torch.stack(gt_image_cache, dim=0)
                        lpips_test += lpips(batch_img, batch_gt_img).sum().double()
                        image_cache = []
                        gt_image_cache = []

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                lpips_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])          
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                    if region_metrics_enabled:
                        for region_key, value in region_metric_sums.items():
                            tb_writer.add_scalar(
                                config['name'] + f"/region - {region_key}",
                                value / len(config['cameras']),
                                iteration,
                            )

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--source_paths', type=str, nargs="*", 
                        help="List of source directories containing images and flame parameters")
    parser.add_argument(
        "--source_sampling_probabilities",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Target sampling probability for each source_path. Values are normalized and applied "
            "per source, so large sources no longer drown out small expression-focused sources."
        ),
    )
    parser.add_argument(
        "--source_regularizer_scales",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Per-source multiplier for deformation regularizers (laplacian, relative_deform, "
            "motion residual L2/laplacian). Defaults to 1 for every source."
        ),
    )
    parser.add_argument('--model_path', type=str, 
                        help="Path to directory where the gaussian avatar model is saved.")
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=10_000, 
                        help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[],
                        help="Extra testing iterations (for visualization).")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--load_existing_checkpoint", type=int, default=None,
                        help="Whether to load existing (newest) checkpoint in model_path")
    parser.add_argument(
        "--init_checkpoint_path",
        type=str,
        default=None,
        help="Initialize model weights from a checkpoint without restoring its optimizer or iteration.",
    )
    parser.add_argument("--config_path", type=str, default = None)
    parser.add_argument("--iterations", type=int, default=None,
                        help="Optional override for opt_params.iterations from the config.")
    parser.add_argument("--lambda_laplacian", type=float, default=None,
                        help="Optional override for opt_params.lambda_laplacian.")
    parser.add_argument("--lambda_relative_deform", type=float, default=None,
                        help="Optional override for opt_params.lambda_relative_deform.")
    parser.add_argument("--lambda_lpips_end", type=float, default=None,
                        help="Optional override for the final LPIPS mixing coefficient.")
    parser.add_argument(
        "--disable_densification",
        action="store_true",
        default=False,
        help="Disable Gaussian densify/prune when fine-tuning from an existing dense checkpoint.",
    )
    parser.add_argument("--enable_pseudo_back", action="store_true", default=False,
                        help="Enable pseudo back-view frames during training.")
    parser.add_argument("--pseudo_back_json", type=str, default=None,
                        help="Path to pseudo_back_frames.json. If omitted, dataset reader will try to auto-discover it.")
    parser.add_argument("--lambda_back_rgb", type=float, default=0.3)
    parser.add_argument("--lambda_back_lpips", type=float, default=0.02)
    parser.add_argument("--lambda_back_sil", type=float, default=0.2)
    parser.add_argument("--back_start_iter", type=int, default=10000)
    parser.add_argument("--back_end_iter", type=int, default=-1)
    parser.add_argument("--back_warmup_iters", type=int, default=10000,
                        help="Linearly ramp pseudo-back sampling and loss weights after back_start_iter.")
    parser.add_argument("--back_sample_ratio", type=float, default=0.10,
                        help="Maximum pseudo-frame sampling probability after warmup.")
    parser.add_argument("--pseudo_back_densify_start_iter", type=int, default=-1,
                        help="Iteration after which pseudo-back views can contribute densification stats. -1 disables it.")
    parser.add_argument("--motion_feature_path", type=str, default=None,
                        help="Optional Xnemo motion feature tensor for joint reconstruction.")
    parser.add_argument("--motion_condition_channels", type=int, default=16,
                        help="Number of projected motion-condition channels appended to the deformation U-Net input in legacy_concat mode.")
    parser.add_argument("--motion_condition_mode", type=str, default="strict_adain",
                        choices=[
                            "legacy_concat",
                            "strict_adain",
                            "strict_adain_bottleneck",
                            "strict_adain_allnorm",
                            "adain",
                            "film",
                            "conditional_norm",
                            "gated_multistage",
                            "residual_branch",
                            "spatial_residual_branch_v2",
                            "cross_attention",
                            "cross_attention_v2",
                            "cross_attention_v3",
                            "cross_attention_v4",
                        ],
                        help="How to inject the 512-dim Xnemo condition into the deformation U-Net.")
    parser.add_argument("--motion_condition_layers", type=str, default="bottleneck",
                        choices=["bottleneck", "gated_multistage", "residual_branch", "allnorm", "spatial_residual_v2", "cross_attention"],
                        help="Which deformation U-Net layer receives conditional normalization.")
    parser.add_argument("--motion_condition_hidden_dim", type=int, default=128,
                        help="Hidden width of the condition-to-affine MLP used by conditional_norm/adain.")
    parser.add_argument("--motion_condition_gamma_scale", type=float, default=0.1,
                        help="Scale applied to condition-generated gamma for stable conditional modulation.")
    parser.add_argument("--motion_condition_norm", type=str, default="existing",
                        choices=["existing", "group", "instance"],
                        help="FiLM baseline norm. Ignored by strict_adain, which explicitly computes spatial mean/std.")
    parser.add_argument("--motion_condition_gate_init", type=float, default=0.05,
                        help="Initial sigmoid gate value for gated_multistage conditional modulation.")
    parser.add_argument("--motion_condition_residual_alpha_init", type=float, default=0.05,
                        help="Initial sigmoid alpha for D_final = D_base + alpha * delta_condition in residual_branch mode.")
    parser.add_argument("--motion_condition_tokens", type=int, default=8,
                        help="Number of Xnemo512-derived condition tokens for cross_attention mode.")
    parser.add_argument("--motion_condition_attention_dim", type=int, default=64,
                        help="Cross-attention query/key/value width for cross_attention mode.")
    parser.add_argument("--motion_cross_attention_gate_init", type=float, default=0.2,
                        help="Initial sigmoid gate value for cross_attention feature residuals.")
    parser.add_argument("--motion_cross_attention_output_init_std", type=float, default=2e-2,
                        help="Initial std for cross_attention output projection weights.")
    parser.add_argument("--motion_cross_attention_logit_scale", type=float, default=4.0,
                        help="Initial cosine-attention logit scale for cross_attention mode.")
    parser.add_argument("--motion_cross_attention_lr_mult", type=float, default=10.0,
                        help="LR multiplier for cross_attention parameters relative to deform_net_lr.")
    parser.add_argument("--motion_cross_attention_w_decay", type=float, default=0.0,
                        help="Weight decay for the cross_attention optimizer parameter group.")
    parser.add_argument("--motion_cross_attention_uv_dropout_prob", type=float, default=0.0,
                        help="Training-only probability of attenuating the UV/expression U-Net input in cross_attention mode.")
    parser.add_argument("--motion_cross_attention_uv_dropout_scale", type=float, default=0.0,
                        help="Multiplier applied to UV/expression input when cross_attention UV dropout fires.")
    parser.add_argument("--motion_cross_attention_uv_noise_std", type=float, default=0.0,
                        help="Training-only Gaussian noise std added to the UV/expression U-Net input in cross_attention mode.")
    parser.add_argument("--motion_cross_attention_base_pretrain_iters", type=int, default=0,
                        help="For cross_attention_v2/v3/v4, train the base U-Net with zero condition for this many iterations.")
    parser.add_argument("--motion_cross_attention_base_lr_mult_after_pretrain", type=float, default=1.0,
                        help="For cross_attention_v2/v3/v4, base U-Net LR multiplier after base pretraining; use 0 to freeze it.")
    parser.add_argument("--motion_cross_attention_condition_warmup_iters", type=int, default=0,
                        help="For cross_attention_v2/v3/v4, linearly warm up cross-attention LR after base pretraining.")
    parser.add_argument("--motion_cross_attention_adapter_only", action="store_true", default=False,
                        help="For cross_attention_v2/v3/v4, freeze all initialized non-attention parameters and disable densification.")
    parser.add_argument(
        "--motion_cross_attention_centering",
        type=str,
        default="training_mean",
        choices=["none", "training_mean"],
        help="For cross_attention_v4, remove the checkpointed training-set common 512 component while preserving exact zero512.",
    )
    parser.add_argument(
        "--motion_cross_attention_mismatch_candidates",
        type=int,
        default=16,
        help="Number of FLAME-nearest candidates considered when selecting a wrong-frame 512 for cross_attention_v4.",
    )
    parser.add_argument(
        "--motion_cross_attention_mismatch_selection",
        type=str,
        default="flame_nearest",
        choices=["random", "flame_nearest"],
        help="Wrong-frame selection policy for cross_attention_v4 causal suppression.",
    )
    parser.add_argument("--lambda_motion_residual_l2", type=float, default=0.0,
                        help="L2 regularization weight for the condition residual branch output.")
    parser.add_argument("--lambda_motion_residual_lap", type=float, default=0.0,
                        help="Laplacian smoothness weight for the condition residual branch output.")
    parser.add_argument("--lambda_motion_residual_ratio", type=float, default=0.0,
                        help="Soft-ceiling loss weight for the actual condition residual RMS divided by frozen-base RMS (cross_attention_v3/v4).")
    parser.add_argument("--motion_residual_ratio_limit", type=float, default=0.35,
                        help="Unpenalized actual condition/base RMS ratio for cross_attention_v3/v4.")
    parser.add_argument(
        "--lambda_motion_mismatch_suppression",
        type=float,
        default=0.0,
        help="For cross_attention_v4, suppress deformation caused by an expression-matched wrong-frame 512 relative to the zero512 base path.",
    )
    parser.add_argument("--motion_feature_align", type=str, default="strict",
                        choices=["strict", "interpolate", "truncate"],
                        help="How to handle a frame-count mismatch between motion features and FLAME timesteps.")
    parser.add_argument("--motion_condition_runtime_shuffle", type=str, default="none",
                        choices=["none", "batch"],
                        help="A2 control: randomly replace the 512 condition per training batch while keeping the same model capacity.")
    parser.add_argument("--motion_nodeform_condition", type=str, default="zero",
                        choices=["zero", "mean", "neutral"],
                        help="Phase-3C nodeform branch condition policy for Xnemo-conditioned runs.")
    parser.add_argument("--condition_monitor_interval", type=int, default=0,
                        help="If >0, log condition sensitivity and AdaIN gamma/beta/gradient stats every N iterations.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for fair short-run comparisons.")
    parser.add_argument("--lambda_region", type=float, default=0.0,
                        help="Global multiplier for Phase-4 region-balanced local supervision.")
    parser.add_argument("--region_w_mouth", type=float, default=1.0)
    parser.add_argument("--region_w_eyes", type=float, default=1.0)
    parser.add_argument("--region_w_brow", type=float, default=1.0)
    parser.add_argument("--region_w_cheeks", type=float, default=0.0)
    parser.add_argument("--region_loss_type", type=str, default="charbonnier", choices=["l1", "charbonnier"])
    parser.add_argument("--region_mask_feather_px", type=float, default=3.0)
    parser.add_argument("--region_mask_mouth_scale", type=float, default=1.18)
    parser.add_argument("--region_mask_eyes_scale", type=float, default=1.20)
    parser.add_argument("--region_mask_brow_scale", type=float, default=1.18)
    parser.add_argument("--region_mask_cheeks_scale", type=float, default=1.08)
    parser.add_argument("--region_mask_min_valid_pixels", type=float, default=8.0)
    parser.add_argument("--region_log_interval", type=int, default=100)
    parser.add_argument("--region_calibration_interval", type=int, default=0,
                        help="If >0, log global-vs-region deformation gradient norms every N iterations.")
    parser.add_argument("--region_calibration_target_ratio", type=float, default=0.3)
    args = parser.parse_args()
    if args.motion_condition_mode == "adain":
        print("WARNING: --motion_condition_mode adain is deprecated; using strict_adain.")
        args.motion_condition_mode = "strict_adain"
    if args.motion_condition_mode == "strict_adain_bottleneck":
        args.motion_condition_mode = "strict_adain"
    if args.motion_condition_mode == "gated_multistage" and args.motion_condition_layers == "bottleneck":
        args.motion_condition_layers = "gated_multistage"
    if args.motion_condition_mode == "residual_branch" and args.motion_condition_layers == "bottleneck":
        args.motion_condition_layers = "residual_branch"
    if args.motion_condition_mode == "strict_adain_allnorm":
        args.motion_condition_layers = "allnorm"
    if args.motion_condition_mode == "spatial_residual_branch_v2":
        args.motion_condition_layers = "spatial_residual_v2"
    if args.motion_condition_mode in (
        "cross_attention",
        "cross_attention_v2",
        "cross_attention_v3",
        "cross_attention_v4",
    ):
        args.motion_condition_layers = "cross_attention"

    args.source_sampling_probabilities = validate_source_vector(
        "source_sampling_probabilities",
        args.source_sampling_probabilities,
        args.source_paths,
        require_positive_sum=True,
    )
    if args.source_sampling_probabilities is not None:
        probability_sum = sum(args.source_sampling_probabilities)
        args.source_sampling_probabilities = [
            value / probability_sum for value in args.source_sampling_probabilities
        ]
    args.source_regularizer_scales = validate_source_vector(
        "source_regularizer_scales",
        args.source_regularizer_scales,
        args.source_paths,
    )

    print("Loading config from", args.config_path)
    configure_cuda_performance()
    config = OmegaConf.load(args.config_path)
    opt_params = config["opt_params"]
    model_params = config["model_params"]
    if args.iterations is not None:
        if args.iterations <= 0:
            raise ValueError("--iterations must be positive when provided.")
        opt_params["iterations"] = int(args.iterations)
    if args.lambda_laplacian is not None:
        if args.lambda_laplacian < 0.0:
            raise ValueError("--lambda_laplacian must be non-negative.")
        opt_params["lambda_laplacian"] = args.lambda_laplacian
    if args.lambda_relative_deform is not None:
        if args.lambda_relative_deform < 0.0:
            raise ValueError("--lambda_relative_deform must be non-negative.")
        opt_params["lambda_relative_deform"] = args.lambda_relative_deform
    if args.lambda_lpips_end is not None:
        if not 0.0 <= args.lambda_lpips_end <= 1.0:
            raise ValueError("--lambda_lpips_end must be in [0, 1].")
        opt_params["lambda_lpips_end"] = args.lambda_lpips_end
    if args.disable_densification:
        opt_params["densify_until_iter"] = 0
    opt_params["lambda_motion_residual_l2"] = args.lambda_motion_residual_l2
    opt_params["lambda_motion_residual_lap"] = args.lambda_motion_residual_lap
    opt_params["lambda_motion_residual_ratio"] = args.lambda_motion_residual_ratio
    opt_params["motion_residual_ratio_limit"] = args.motion_residual_ratio_limit
    opt_params["lambda_motion_mismatch_suppression"] = (
        args.lambda_motion_mismatch_suppression
    )
    opt_params["lambda_region"] = args.lambda_region
    opt_params["region_w_mouth"] = args.region_w_mouth
    opt_params["region_w_eyes"] = args.region_w_eyes
    opt_params["region_w_brow"] = args.region_w_brow
    opt_params["region_w_cheeks"] = args.region_w_cheeks
    opt_params["region_loss_type"] = args.region_loss_type
    opt_params["region_mask_feather_px"] = args.region_mask_feather_px
    opt_params["region_mask_mouth_scale"] = args.region_mask_mouth_scale
    opt_params["region_mask_eyes_scale"] = args.region_mask_eyes_scale
    opt_params["region_mask_brow_scale"] = args.region_mask_brow_scale
    opt_params["region_mask_cheeks_scale"] = args.region_mask_cheeks_scale
    opt_params["region_mask_min_valid_pixels"] = args.region_mask_min_valid_pixels
    opt_params["region_log_interval"] = args.region_log_interval
    opt_params["region_calibration_interval"] = args.region_calibration_interval
    opt_params["region_calibration_target_ratio"] = args.region_calibration_target_ratio
    opt_params["seed"] = args.seed
    opt_params["init_checkpoint_path"] = args.init_checkpoint_path
    opt_params["source_sampling_probabilities"] = args.source_sampling_probabilities
    opt_params["source_regularizer_scales"] = args.source_regularizer_scales

    if args.init_checkpoint_path is not None and args.load_existing_checkpoint:
        raise ValueError("Use either --init_checkpoint_path or --load_existing_checkpoint, not both.")
    if (
        args.init_checkpoint_path is not None
        and not args.motion_cross_attention_adapter_only
        and not args.disable_densification
    ):
        print(
            "WARNING: fine-tuning from a dense checkpoint with densification enabled can grow the "
            "Gaussian set again; pass --disable_densification unless this is intentional."
        )
    if args.motion_cross_attention_adapter_only:
        if args.motion_condition_mode not in (
            "cross_attention_v2",
            "cross_attention_v3",
            "cross_attention_v4",
        ):
            raise ValueError(
                "--motion_cross_attention_adapter_only requires --motion_condition_mode "
                "cross_attention_v2, cross_attention_v3, or cross_attention_v4."
            )
        if args.init_checkpoint_path is None:
            raise ValueError("--motion_cross_attention_adapter_only requires --init_checkpoint_path.")
        if args.motion_cross_attention_base_pretrain_iters != 0:
            raise ValueError("Adapter-only initialization requires --motion_cross_attention_base_pretrain_iters 0.")
    if args.lambda_motion_residual_ratio < 0.0:
        raise ValueError("--lambda_motion_residual_ratio must be non-negative.")
    if args.motion_residual_ratio_limit < 0.0:
        raise ValueError("--motion_residual_ratio_limit must be non-negative.")
    if (
        args.lambda_motion_residual_ratio > 0.0
        and args.motion_condition_mode not in ("cross_attention_v3", "cross_attention_v4")
    ):
        raise ValueError(
            "--lambda_motion_residual_ratio currently requires --motion_condition_mode "
            "cross_attention_v3 or cross_attention_v4."
        )
    if args.lambda_motion_mismatch_suppression < 0.0:
        raise ValueError("--lambda_motion_mismatch_suppression must be non-negative.")
    if args.motion_cross_attention_mismatch_candidates < 1:
        raise ValueError("--motion_cross_attention_mismatch_candidates must be at least 1.")
    if (
        args.lambda_motion_mismatch_suppression > 0.0
        and args.motion_condition_mode != "cross_attention_v4"
    ):
        raise ValueError(
            "--lambda_motion_mismatch_suppression requires --motion_condition_mode "
            "cross_attention_v4."
        )
    if args.motion_feature_path is not None:
        validate_motion_feature_index(args.source_paths, args.motion_feature_path)
        model_params["use_motion_condition"] = True
        model_params["motion_feature_path"] = args.motion_feature_path
        model_params["motion_condition_channels"] = args.motion_condition_channels
        model_params["motion_condition_mode"] = args.motion_condition_mode
        model_params["motion_condition_layers"] = args.motion_condition_layers
        model_params["motion_condition_hidden_dim"] = args.motion_condition_hidden_dim
        model_params["motion_condition_gamma_scale"] = args.motion_condition_gamma_scale
        model_params["motion_condition_norm"] = args.motion_condition_norm
        model_params["motion_condition_gate_init"] = args.motion_condition_gate_init
        model_params["motion_condition_residual_alpha_init"] = args.motion_condition_residual_alpha_init
        model_params["motion_condition_tokens"] = args.motion_condition_tokens
        model_params["motion_condition_attention_dim"] = args.motion_condition_attention_dim
        model_params["motion_cross_attention_gate_init"] = args.motion_cross_attention_gate_init
        model_params["motion_cross_attention_output_init_std"] = args.motion_cross_attention_output_init_std
        model_params["motion_cross_attention_logit_scale"] = args.motion_cross_attention_logit_scale
        model_params["motion_cross_attention_lr_mult"] = args.motion_cross_attention_lr_mult
        model_params["motion_cross_attention_w_decay"] = args.motion_cross_attention_w_decay
        model_params["motion_cross_attention_uv_dropout_prob"] = args.motion_cross_attention_uv_dropout_prob
        model_params["motion_cross_attention_uv_dropout_scale"] = args.motion_cross_attention_uv_dropout_scale
        model_params["motion_cross_attention_uv_noise_std"] = args.motion_cross_attention_uv_noise_std
        model_params["motion_cross_attention_base_pretrain_iters"] = args.motion_cross_attention_base_pretrain_iters
        model_params["motion_cross_attention_base_lr_mult_after_pretrain"] = args.motion_cross_attention_base_lr_mult_after_pretrain
        model_params["motion_cross_attention_condition_warmup_iters"] = args.motion_cross_attention_condition_warmup_iters
        model_params["motion_cross_attention_adapter_only"] = args.motion_cross_attention_adapter_only
        model_params["motion_cross_attention_centering"] = args.motion_cross_attention_centering
        model_params["motion_cross_attention_mismatch_enabled"] = (
            args.lambda_motion_mismatch_suppression > 0.0
        )
        model_params["motion_cross_attention_mismatch_candidates"] = (
            args.motion_cross_attention_mismatch_candidates
        )
        model_params["motion_cross_attention_mismatch_selection"] = (
            args.motion_cross_attention_mismatch_selection
        )
        model_params["motion_feature_align"] = args.motion_feature_align
        model_params["motion_condition_runtime_shuffle"] = args.motion_condition_runtime_shuffle
        model_params["motion_nodeform_condition"] = args.motion_nodeform_condition

    if args.interval > opt_params["iterations"]:
        args.interval = opt_params["iterations"] // 5
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, opt_params["iterations"]+1, args.interval)))
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, opt_params["iterations"]+1, args.interval)))

    print("Optimizing " + args.model_path)
    model_path = Path(args.model_path)
    model_path.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(config, model_path / "config_dump.yaml")

    # Initialize system state (RNG)
    safe_state(args.quiet)
    if args.seed != 0:
        set_random_seed(args.seed)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        args.source_paths,
        args.model_path,
        model_params,
        opt_params,
        args.test_iterations, 
        args.checkpoint_iterations, 
        args.load_existing_checkpoint, 
        args.init_checkpoint_path,
        args.enable_pseudo_back,
        args.pseudo_back_json,
        args.lambda_back_rgb,
        args.lambda_back_lpips,
        args.lambda_back_sil,
        args.back_start_iter,
        args.back_end_iter,
        args.back_warmup_iters,
        args.back_sample_ratio,
        args.pseudo_back_densify_start_iter,
        args.lambda_motion_residual_l2,
        args.lambda_motion_residual_lap,
        args.lambda_motion_residual_ratio,
        args.motion_residual_ratio_limit,
        args.lambda_motion_mismatch_suppression,
        args.condition_monitor_interval,
        args.source_sampling_probabilities,
        args.source_regularizer_scales,
    )

    # All done
    print("\nTraining complete.")
