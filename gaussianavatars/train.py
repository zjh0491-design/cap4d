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
import shutil
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image

from gaussianavatars.utils.system_utils import searchForMaxIteration
from gaussianavatars.gaussian_renderer.gsplat_renderer import render
from gaussianavatars.scene.cap4d_gaussian_model import CAP4DGaussianModel
from gaussianavatars.scene.scene import CameraDataset, Scene
from gaussianavatars.utils.loss_utils import l1_loss, ssim
from gaussianavatars.utils.general_utils import safe_state
from gaussianavatars.utils.image_utils import psnr, error_map
from gaussianavatars.lpipsPyTorch import LPIPS
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def next_camera_from_loader(loader, loader_iter):
    try:
        camera = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        camera = next(loader_iter)
    return camera, loader_iter


def masked_l1_loss(network_output, gt, mask):
    denom = mask.sum().clamp_min(1.) * network_output.shape[0]
    return (torch.abs(network_output - gt) * mask).sum() / denom


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


def training(
    source_paths,
    model_path,
    model_params, 
    opt_params, 
    testing_iterations, 
    checkpoint_iterations, 
    load_existing_checkpoint, 
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

        loader_camera_real = DataLoader(
            CameraDataset(real_train_cameras),
            batch_size=None,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
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
        loader_camera_train = DataLoader(
            scene.getTrainCameras(),
            batch_size=None,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )
        iter_camera_train = iter(loader_camera_train)

    ema_loss_for_log = 0.0
    pseudo_sample_count = 0
    progress_bar = tqdm(range(first_iter, opt_params["iterations"]), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt_params["iterations"] + 1):    
        iter_start.record()

        gaussians.train()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % opt_params["sh_warmup_iterations"] == 0:
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

        lambda_lpips = 0.
        is_pseudo_view = getattr(viewpoint_cam, "is_pseudo", False)

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
            losses['lap'] = gaussians.compute_laplacian_loss() * opt_params["lambda_laplacian"]

        if opt_params["lambda_relative_deform"] != 0:
            losses['deform'] = gaussians.compute_relative_deformation_loss() * opt_params["lambda_relative_deform"]

        if opt_params["lambda_relative_rot"] != 0:
            losses['rot'] = gaussians.compute_relative_rotation_loss() * opt_params["lambda_relative_rot"]

        if opt_params["lambda_neck"] != 0:
            losses['neck'] = gaussians.compute_neck_loss() * opt_params["lambda_neck"]
        
        log_only_losses = {"real_rgb"}
        losses['total'] = sum([v for k, v in losses.items() if k not in log_only_losses])
        losses['total'].backward()

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
                {"pseudo_sample_count": pseudo_sample_count},
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
            if iteration < opt_params["densify_until_iter"]:
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
        if 'deform' in losses:
            tb_writer.add_scalar('train_loss_patches/deform', losses['deform'].detach().item(), iteration)
        if 'rot' in losses:
            tb_writer.add_scalar('train_loss_patches/rot', losses['rot'].detach().item(), iteration)
        if 'dynamic_offset_std' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset_std', losses['dynamic_offset_std'].detach().item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', losses['total'].detach().item(), iteration)
        tb_writer.add_scalar('train/pseudo_sample_count', extra_logs.get("pseudo_sample_count", 0), iteration)
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

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--source_paths', type=str, nargs="*", 
                        help="List of source directories containing images and flame parameters")
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
    parser.add_argument("--config_path", type=str, default = None)
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
    args = parser.parse_args()

    print("Loading config from", args.config_path)
    config = OmegaConf.load(args.config_path)
    opt_params = config["opt_params"]
    model_params = config["model_params"]

    if args.interval > opt_params["iterations"]:
        args.interval = opt_params["iterations"] // 5
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, opt_params["iterations"]+1, args.interval)))
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, opt_params["iterations"]+1, args.interval)))

    print("Optimizing " + args.model_path)
    model_path = Path(args.model_path)
    model_path.mkdir(exist_ok=True, parents=True)
    shutil.copy(args.config_path, model_path / "config_dump.yaml")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        args.source_paths,
        args.model_path,
        model_params,
        opt_params,
        args.test_iterations, 
        args.checkpoint_iterations, 
        args.load_existing_checkpoint, 
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
    )

    # All done
    print("\nTraining complete.")
