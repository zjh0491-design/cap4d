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
from pathlib import Path
from os import makedirs
import subprocess
import concurrent.futures

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np
from omegaconf import OmegaConf

from argparse import ArgumentParser
from gaussianavatars.utils.system_utils import searchForMaxIteration
from gaussianavatars.utils.general_utils import safe_state
from gaussianavatars.gaussian_renderer.gsplat_renderer import render
from gaussianavatars.scene.scene import Scene
from gaussianavatars.scene.cap4d_gaussian_model_xnemo import CAP4DGaussianModel
from gaussianavatars.utils.export_utils import PlyWriter


def write_data(path2data):
    for path, data in path2data.items():
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            if data.dtype == torch.long:
                data = data[0].to("cpu", torch.uint16).numpy()
            else:
                data = data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")


def frames_to_video(
    frame_dir,
    output_path,
    fps,
):
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-f", "image2",
        "-pattern_type", "glob",
        "-i", f"{frame_dir}/*.png",
        "-crf", "18",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        f"{output_path}"
    ]

    # Run the command with real-time output
    subprocess.run(cmd, check=True)


def render_sequence(args):
    model_path = Path(args.model_path)
    target_paths = {
        "animation_path": args.target_animation_path,
        "cam_trajectory_path": args.target_cam_trajectory_path,
    }

    avatar_config = OmegaConf.load(model_path / "config_dump.yaml")
    uses_motion_condition = avatar_config["model_params"].get("use_motion_condition", False)
    if not uses_motion_condition and args.target_motion_feature_path is not None:
        raise ValueError(
            "--target_motion_feature_path was provided, but this avatar was not "
            "trained with Xnemo motion conditioning. Use an avatar trained by "
            "gaussianavatars/train_xnemo.py with --motion_feature_path, or omit "
            "--target_motion_feature_path."
        )
    if uses_motion_condition:
        avatar_config["model_params"]["motion_feature_path"] = args.target_motion_feature_path
        if args.target_motion_feature_path is None:
            raise ValueError(
                "This avatar was trained with Xnemo motion conditioning, so "
                "--target_motion_feature_path is required for animation."
            )

    gaussians = CAP4DGaussianModel(avatar_config["model_params"])
    gaussians.eval()
    
    scene = Scene(
        source_paths=None,
        target_paths=target_paths,
        model_path=model_path,
        gaussians=gaussians, 
        shuffle=False,
    )

    if args.iteration is not None and args.iteration >= 0:
        loaded_iter = int(args.iteration)
        chkpt_path = model_path / f"chkpnt{loaded_iter}.pth"
        assert chkpt_path.exists(), f"No checkpoint found at requested iteration: {chkpt_path}"
    else:
        loaded_iter, chkpt_path = searchForMaxIteration(model_path)
        assert loaded_iter is not None, f"No valid checkpoint found in {model_path}"
    print("Loading trained model at iteration {}".format(loaded_iter))
    (model_weights, first_iter) = torch.load(chkpt_path, weights_only=False)
    gaussians.restore(model_weights)
    if uses_motion_condition:
        motion_features = gaussians.motion_features
        if motion_features is None:
            raise RuntimeError("Motion-conditioned animation has no loaded target motion features.")
        if int(motion_features.shape[0]) != int(gaussians.num_timesteps):
            raise RuntimeError(
                "Target motion feature count changed or was overwritten during restore: "
                f"features={motion_features.shape[0]}, target_timesteps={gaussians.num_timesteps}"
            )
        sample_indices = sorted({0, int(motion_features.shape[0] // 2), int(motion_features.shape[0] - 1)})
        sample_norms = {
            int(index): float(motion_features[index].detach().float().norm().cpu())
            for index in sample_indices
        }
        temporal_delta = (
            float((motion_features[1:] - motion_features[:-1]).detach().float().norm(dim=1).mean().cpu())
            if motion_features.shape[0] > 1 else 0.0
        )
        print(
            "Animation motion condition audit:",
            f"mode={gaussians.motion_condition_mode}",
            f"path={args.target_motion_feature_path}",
            f"shape={tuple(motion_features.shape)}",
            f"target_timesteps={gaussians.num_timesteps}",
            f"sample_norms={sample_norms}",
            f"temporal_delta_mean={temporal_delta:.6f}",
        )
        if gaussians.motion_condition_mode == "cross_attention_v4":
            centered_features = gaussians._condition_motion_feature(motion_features)
            print(
                "Animation cross_attention_v4 preprocessing audit:",
                f"center_source={gaussians.motion_feature_center_source}",
                f"training_common_energy_ratio={gaussians.motion_feature_common_energy_ratio:.6f}",
                f"training_centered_rms={gaussians.motion_feature_centered_rms:.6f}",
                f"target_centered_rms={float(centered_features.detach().float().square().mean().sqrt().cpu()):.6f}",
            )

    bg_color = [1, 1, 1]  # force white background
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    output_path = Path(args.output_path)
    makedirs(output_path, exist_ok=True)
    render_path = output_path / "renders"
    if args.render_alpha:
        render_alpha_path = output_path / "renders_alpha"
        makedirs(render_alpha_path, exist_ok=True)
    if args.render_depth:
        render_depth_path = output_path / "renders_depth"
        makedirs(render_depth_path, exist_ok=True)
    if args.export_ply:
        ply_writer = PlyWriter(compress=args.compress_ply)

    makedirs(render_path, exist_ok=True)

    views_loader = DataLoader(
        scene.getTgtCameras(), 
        batch_size=None, 
        shuffle=False, 
        num_workers=8,
    )
    max_threads = 4
    worker_args = []

    for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):

        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(view.timestep)

        render_out = render(
            view, 
            gaussians, 
            background, 
            compute_depth=args.render_depth
        )

        if uses_motion_condition and int(view.timestep) in sample_indices:
            actual_feature = getattr(gaussians, "_last_prepared_motion_feature", None)
            if actual_feature is None:
                raise RuntimeError("The deformation U-Net did not record a motion condition during rendering.")
            expected_feature = motion_features[[int(view.timestep)]].to(
                device=actual_feature.device,
                dtype=actual_feature.dtype,
            )
            feature_diff = float((actual_feature - expected_feature).abs().max().detach().cpu())
            if feature_diff != 0.0:
                raise RuntimeError(
                    "Rendered motion feature differs from the target timestep feature: "
                    f"timestep={int(view.timestep)}, max_abs_diff={feature_diff:.6e}"
                )
            conditioned_feature_diff = 0.0
            conditioned_norm = 0.0
            if gaussians.motion_condition_mode == "cross_attention_v4":
                actual_conditioned_feature = getattr(
                    gaussians,
                    "_last_condition_motion_feature",
                    None,
                )
                if actual_conditioned_feature is None:
                    raise RuntimeError(
                        "cross_attention_v4 did not record its conditioned motion feature."
                    )
                expected_conditioned_feature = gaussians._condition_motion_feature(
                    expected_feature
                )
                conditioned_feature_diff = float(
                    (actual_conditioned_feature - expected_conditioned_feature)
                    .abs()
                    .max()
                    .detach()
                    .cpu()
                )
                if conditioned_feature_diff != 0.0:
                    raise RuntimeError(
                        "Rendered conditioned feature differs from the checkpoint-centered "
                        "target feature: "
                        f"timestep={int(view.timestep)}, "
                        f"max_abs_diff={conditioned_feature_diff:.6e}"
                    )
                conditioned_norm = float(
                    actual_conditioned_feature.float().norm().detach().cpu()
                )
            print(
                "Rendered motion condition verified:",
                f"timestep={int(view.timestep)}",
                f"norm={float(actual_feature.float().norm().detach().cpu()):.6f}",
                f"max_abs_diff={feature_diff:.1e}",
                f"conditioned_norm={conditioned_norm:.6f}",
                f"conditioned_max_abs_diff={conditioned_feature_diff:.1e}",
            )
        
        rendering = render_out["render"]

        if args.export_ply:
            ply_writer.update(gaussians)

        path2data = {}
        path2data[Path(render_path) / f'{idx:05d}.png'] = rendering
        if args.render_alpha:
            alpha = render_out["alpha"]
            blended_rendering = torch.cat([rendering, alpha], dim=0)
            path2data[Path(render_alpha_path) / f'{idx:05d}.png'] = blended_rendering
        if args.render_depth:
            depth = render_out["depth"]
            depth = (depth * 1000.).long()
            path2data[Path(render_depth_path) / f'{idx:05d}.png'] = depth

        worker_args.append([path2data])

        if len(worker_args) == max_threads or idx == len(views_loader)-1:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(write_data, *args) for args in worker_args]
                concurrent.futures.wait(futures)
            worker_args = []

    if args.export_ply:
        print("Exporting animation...")
        ply_writer.save_ply(output_path / "exported_animation.ply")

    # load fps
    fps = 24  # default fps
    if args.target_cam_trajectory_path is not None:
        camera_trajectory = dict(np.load(args.target_cam_trajectory_path))
        if "fps" in camera_trajectory:
            fps = camera_trajectory["fps"]

    frames_to_video(render_path, output_path / "renders.mp4", fps=fps)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Render the reconstructed avatar with a " \
    "driving animation and camera trajectory.")
    parser.add_argument('--model_path', type=str, help="Path to directory " \
                                            "where the gaussian avatar model is saved.")
    parser.add_argument('--target_animation_path', type=str, help="Path to driving animation (fit.npz).")
    parser.add_argument('--target_cam_trajectory_path', type=str, default=None,
                        help="Path to driving camera trajectory (*.npz). " \
                        "This trajectory describes per frame camera intrinsics and " \
                        "extrinsics - the extrinsics are expressed relative to the " \
                        "extrinsics of the input animation. If this is not given," \
                        "the camera of the driving sequence will be used.")
    parser.add_argument("--target_motion_feature_path", type=str, default=None,
                        help="Optional Xnemo motion feature tensor aligned with the target animation.")
    parser.add_argument('--output_path', type=str, required=True,
                        help="Path to directory where the animation outputs " \
                        "(frames, video and optionally ply) will be saved.")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_alpha", type=int, default=False)
    parser.add_argument("--render_depth", type=int, default=False)
    parser.add_argument("--export_ply", type=int, default=True, 
                        help="save baked ply animation for web rendering")
    parser.add_argument("--compress_ply", type=int, default=False, 
                        help="compress baked ply animation at the cost of quality (jittering)")

    args = parser.parse_args()
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    with torch.no_grad():
        render_sequence(args)
