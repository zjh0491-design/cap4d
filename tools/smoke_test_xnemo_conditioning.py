from argparse import ArgumentParser
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussianavatars.scene.net.unet import (
    ConditionalAffine2d,
    ConditionalInstanceNorm2d,
    ConditionalLayerNorm,
    ConditionalResidualBranch,
    CrossAttentionCondition2d,
    GatedConditionalAffine2d,
    SharedStrictAdaIN2d,
    SpatialResidualBranchV2,
    define_G,
)


def assert_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or Inf")


def perturb_condition_affine(model, std, require_count=True):
    count = 0
    for module in model.modules():
        if isinstance(module, (ConditionalAffine2d, ConditionalInstanceNorm2d, ConditionalLayerNorm)):
            with torch.no_grad():
                torch.nn.init.normal_(module.mlp[-1].weight, mean=0.0, std=std)
                torch.nn.init.zeros_(module.mlp[-1].bias)
            count += 1
        elif isinstance(module, SharedStrictAdaIN2d):
            with torch.no_grad():
                torch.nn.init.normal_(module.head.weight, mean=0.0, std=std)
                torch.nn.init.zeros_(module.head.bias)
            count += 1
    if require_count and count == 0:
        raise RuntimeError("No ConditionalAffine2d module found in conditional U-Net")
    return count


def count_strict_adain(model):
    return sum(1 for module in model.modules() if isinstance(module, ConditionalInstanceNorm2d))


def count_gates(model):
    return sum(1 for module in model.modules() if isinstance(module, GatedConditionalAffine2d))


def count_residual_branches(model):
    return sum(1 for module in model.modules() if isinstance(module, ConditionalResidualBranch))


def count_allnorm_adain(model):
    return sum(1 for module in model.modules() if isinstance(module, SharedStrictAdaIN2d))


def count_spatial_residual_v2(model):
    return sum(1 for module in model.modules() if isinstance(module, SpatialResidualBranchV2))


def count_cross_attention(model):
    return sum(1 for module in model.modules() if isinstance(module, CrossAttentionCondition2d))


def grad_norm(parameters):
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += float((grad * grad).sum().cpu())
    return total ** 0.5


def condition_grad_norm(model):
    total = 0.0
    for module in model.modules():
        if isinstance(
            module,
            (
                ConditionalAffine2d,
                ConditionalInstanceNorm2d,
                ConditionalLayerNorm,
                GatedConditionalAffine2d,
                ConditionalResidualBranch,
                SharedStrictAdaIN2d,
                SpatialResidualBranchV2,
                CrossAttentionCondition2d,
            ),
        ):
            total += grad_norm(module.parameters()) ** 2
    return total ** 0.5


def verify_strict_normalizers(device):
    condition = torch.randn(2, 512, device=device)
    spatial_norm = ConditionalInstanceNorm2d(8).to(device)
    token_norm = ConditionalLayerNorm(8).to(device)

    spatial_x = torch.randn(2, 8, 4, 5, device=device) * 3.0 + 2.0
    token_x = torch.randn(2, 6, 8, device=device) * 3.0 + 2.0
    with torch.no_grad():
        spatial_y = spatial_norm(spatial_x, condition)
        token_y = token_norm(token_x, condition)

    spatial_mean = spatial_y.mean(dim=(2, 3)).abs().max().item()
    spatial_std = spatial_y.var(dim=(2, 3), unbiased=False).add(1e-5).sqrt().sub(1.).abs().max().item()
    token_mean = token_y.mean(dim=-1).abs().max().item()
    token_std = token_y.var(dim=-1, unbiased=False).add(1e-5).sqrt().sub(1.).abs().max().item()
    if max(spatial_mean, spatial_std, token_mean, token_std) > 1e-4:
        raise RuntimeError(
            "Strict conditional normalizers failed zero-affine normalization check: "
            f"spatial_mean={spatial_mean:.3e}, spatial_std={spatial_std:.3e}, "
            f"token_mean={token_mean:.3e}, token_std={token_std:.3e}"
        )
    return spatial_mean, spatial_std, token_mean, token_std


def make_condition_layers(condition_mode, requested_layers):
    if requested_layers is not None:
        return requested_layers
    if condition_mode == "gated_multistage":
        return "gated_multistage"
    if condition_mode == "residual_branch":
        return "residual_branch"
    if condition_mode == "strict_adain_allnorm":
        return "allnorm"
    if condition_mode == "spatial_residual_branch_v2":
        return "spatial_residual_v2"
    if condition_mode in (
        "cross_attention",
        "cross_attention_v2",
        "cross_attention_v3",
        "cross_attention_v4",
    ):
        return "cross_attention"
    return "bottleneck"


def run_baseline_check(args, device):
    h = w = args.uv_resolution
    baseline_input_channels = 3 + args.pos_encoding_channels
    baseline_net = define_G(
        baseline_input_channels,
        3,
        args.ngf,
        f"unet_{args.uv_resolution}",
        n_layers=args.n_unet_layers,
        norm="instance",
        condition_mode="legacy_concat",
    ).to(device)
    baseline_input = torch.randn(2, baseline_input_channels, h, w, device=device, requires_grad=True)
    baseline_output = baseline_net(baseline_input)
    assert baseline_output.shape == (2, 3, h, w)
    assert_finite("baseline_output", baseline_output)
    baseline_loss = baseline_output.square().mean()
    baseline_loss.backward()
    if grad_norm(baseline_net.parameters()) <= 0.0:
        raise RuntimeError("Baseline U-Net backward produced zero parameter gradients.")
    return {
        "baseline_output_shape": tuple(baseline_output.shape),
        "baseline_grad_norm": grad_norm(baseline_net.parameters()),
    }


def run_legacy_concat_check(args, device):
    h = w = args.uv_resolution
    legacy_input_channels = 3 + args.pos_encoding_channels + args.motion_condition_channels
    legacy_net = define_G(
        legacy_input_channels,
        3,
        args.ngf,
        f"unet_{args.uv_resolution}",
        n_layers=args.n_unet_layers,
        norm="instance",
        condition_mode="legacy_concat",
    ).to(device)
    legacy_input = torch.randn(2, legacy_input_channels, h, w, device=device, requires_grad=True)
    legacy_output = legacy_net(legacy_input)
    assert legacy_output.shape == (2, 3, h, w)
    assert_finite("legacy_output", legacy_output)
    legacy_output.square().mean().backward()
    return {
        "legacy_output_shape": tuple(legacy_output.shape),
        "legacy_grad_norm": grad_norm(legacy_net.parameters()),
    }


def perturb_condition_path(model, std):
    count = perturb_condition_affine(model, std, require_count=False)
    for module in model.modules():
        if isinstance(module, SpatialResidualBranchV2):
            with torch.no_grad():
                torch.nn.init.normal_(module.scale_1x.condition_head.weight, mean=0.0, std=std)
                torch.nn.init.normal_(module.scale_2x.condition_head.weight, mean=0.0, std=std)
                torch.nn.init.zeros_(module.scale_1x.condition_head.bias)
                torch.nn.init.zeros_(module.scale_2x.condition_head.bias)
                torch.nn.init.normal_(module.fusion[-1].weight, mean=0.0, std=std)
                if module.fusion[-1].bias is not None:
                    torch.nn.init.zeros_(module.fusion[-1].bias)
            count += 1
        elif isinstance(module, CrossAttentionCondition2d):
            with torch.no_grad():
                probe_std = max(float(std), float(module.output_init_std))
                torch.nn.init.normal_(module.out_proj.weight, mean=0.0, std=probe_std)
                if module.out_proj.bias is not None:
                    torch.nn.init.zeros_(module.out_proj.bias)
            count += 1
    if count == 0:
        raise RuntimeError("No condition path found in conditional U-Net")
    return count


def run_conditional_mode(args, device, condition_mode):
    h = w = args.uv_resolution
    input_channels = 3 + args.pos_encoding_channels
    condition_layers = make_condition_layers(condition_mode, args.condition_layers)
    net = define_G(
        input_channels,
        3,
        args.ngf,
        f"unet_{args.uv_resolution}",
        n_layers=args.n_unet_layers,
        norm="instance",
        condition_mode=condition_mode,
        condition_dim=512,
        condition_layers=condition_layers,
        condition_hidden_dim=args.condition_hidden_dim,
        condition_gamma_scale=args.condition_gamma_scale,
        condition_norm=args.condition_norm,
        condition_gate_init=args.condition_gate_init,
        condition_residual_alpha_init=args.condition_residual_alpha_init,
        condition_num_tokens=args.condition_tokens,
        condition_attention_dim=args.condition_attention_dim,
        condition_cross_attention_gate_init=args.cross_attention_gate_init,
        condition_attention_output_init_std=args.cross_attention_output_init_std,
        condition_attention_logit_scale=args.cross_attention_logit_scale,
    ).to(device)
    net.train()
    adain_input = torch.randn(2, input_channels, h, w, device=device, requires_grad=True)
    condition_a = torch.randn(2, 512, device=device)
    condition_b = torch.randn(2, 512, device=device)

    output = net(adain_input, condition=condition_a)
    assert output.shape == (2, 3, h, w)
    assert_finite(f"{condition_mode}_output", output)
    loss = output.square().mean()
    loss.backward()
    total_grad = grad_norm(net.parameters())
    cond_grad = condition_grad_norm(net)
    if total_grad <= 0.0:
        raise RuntimeError(f"{condition_mode} backward produced zero total gradients.")
    if cond_grad <= 0.0:
        raise RuntimeError(f"{condition_mode} backward produced zero condition-path gradients.")

    net.eval()
    with torch.no_grad():
        base_output = net(adain_input.detach(), condition=condition_a)
        zero_condition = torch.zeros_like(condition_a)
        zero_output_before_probe = net(adain_input.detach(), condition=zero_condition)
        adain_input_changed = adain_input.detach().clone()
        adain_input_changed[:, :3] = adain_input_changed[:, :3] + 0.1 * torch.randn_like(adain_input_changed[:, :3])
        spatial_changed_output = net(adain_input_changed, condition=condition_a)

    condition_residual = net.get_condition_residual()
    if condition_mode in ("residual_branch", "spatial_residual_branch_v2", "cross_attention_v2"):
        if condition_residual is None:
            raise RuntimeError(f"{condition_mode} did not expose a condition residual.")
        if condition_residual.shape != base_output.shape:
            raise RuntimeError(
                "Condition residual shape does not match U-Net output: "
                f"residual={tuple(condition_residual.shape)}, output={tuple(base_output.shape)}"
            )
        assert_finite("condition_residual", condition_residual)
    elif condition_residual is not None:
        raise RuntimeError("Non-residual condition mode unexpectedly exposed a condition residual.")
    spatial_diff = (base_output - spatial_changed_output).abs().max().item()
    if spatial_diff <= args.min_spatial_diff:
        raise RuntimeError(
            "Conditional U-Net output did not change when the UV/expression input changed: "
            f"max_diff={spatial_diff:.6e}"
        )

    modulated_layers = perturb_condition_path(net, args.condition_probe_std)
    with torch.no_grad():
        output_a = net(adain_input.detach(), condition=condition_a)
        output_b = net(adain_input.detach(), condition=condition_b)
        zero_output_after_probe = net(adain_input.detach(), condition=zero_condition)
    assert_finite("output_a", output_a)
    assert_finite("output_b", output_b)
    condition_diff = (output_a - output_b).abs().max().item()
    if condition_diff <= args.min_condition_diff:
        raise RuntimeError(
            "Conditional U-Net output did not change under different 512-dim conditions: "
            f"max_diff={condition_diff:.6e}"
        )
    zero_identity_diff = (zero_output_before_probe - zero_output_after_probe).abs().max().item()
    if condition_mode in (
        "cross_attention",
        "cross_attention_v2",
        "cross_attention_v3",
        "cross_attention_v4",
    ) and zero_identity_diff != 0.0:
        raise RuntimeError(
            "Cross-attention changed its output under an exact zero condition: "
            f"max_diff={zero_identity_diff:.6e}"
        )

    deform_input = torch.randn(1, input_channels, h, w, device=device)
    nodeform_input = torch.randn(1, input_channels, h, w, device=device)
    nodeform_input[:, :3] = 0.0
    unet_input = torch.cat([deform_input, nodeform_input], dim=0)
    deform_condition = torch.randn(1, 512, device=device)
    nodeform_condition = torch.zeros_like(deform_condition)
    condition_input = torch.cat([deform_condition, nodeform_condition], dim=0)
    if condition_input.shape[0] != unet_input.shape[0]:
        raise RuntimeError("deform/nodeform condition batch concatenation mismatch.")
    with torch.no_grad():
        dual_output = net(unet_input, condition=condition_input)
    deform_output, nodeform_output = dual_output.chunk(2, dim=0)
    if deform_output.shape != nodeform_output.shape:
        raise RuntimeError("deform/nodeform output split shape mismatch.")

    info = {
        "condition_mode": condition_mode,
        "condition_layers": condition_layers,
        "conditional_output_shape": tuple(base_output.shape),
        "conditional_layers": modulated_layers,
        "strict_adain_layers": count_strict_adain(net),
        "strict_adain_allnorm_layers": count_allnorm_adain(net),
        "gated_layers": count_gates(net),
        "residual_branches": count_residual_branches(net),
        "spatial_residual_v2_branches": count_spatial_residual_v2(net),
        "cross_attention_layers": count_cross_attention(net),
        "spatial_diff_max": spatial_diff,
        "fixed_uv_condition_diff_max_after_probe": condition_diff,
        "zero_condition_identity_diff_max": zero_identity_diff,
        "total_grad_norm": total_grad,
        "condition_grad_norm": cond_grad,
        "dual_unet_input_shape": tuple(unet_input.shape),
        "dual_condition_shape": tuple(condition_input.shape),
        "dual_deform_output_shape": tuple(deform_output.shape),
        "dual_nodeform_output_shape": tuple(nodeform_output.shape),
    }
    if hasattr(net, "get_adain_site_table"):
        info["adain_site_table"] = net.get_adain_site_table()
    if hasattr(net, "get_cross_attention_site_table"):
        cross_attention_site_table = net.get_cross_attention_site_table()
        info["cross_attention_site_table"] = cross_attention_site_table
        info["cross_attention_param_count"] = sum(
            row.get("parameters", 0) for row in cross_attention_site_table
        )
        if condition_mode in ("cross_attention_v3", "cross_attention_v4"):
            sites = [row["site"] for row in cross_attention_site_table]
            required_suffixes = (
                "decoder_cross_attention_c256",
                "decoder_cross_attention_c128",
                "decoder_cross_attention_c64",
            )
            if len(sites) != 3 or any("output_cross_attention_c3" in site for site in sites):
                raise RuntimeError(
                    f"{condition_mode} must have exactly three feature-only attention sites: "
                    f"sites={sites}"
                )
            if not all(any(site.endswith(suffix) for site in sites) for suffix in required_suffixes):
                raise RuntimeError(
                    f"{condition_mode} is missing a required decoder feature site: "
                    f"sites={sites}"
                )
    if hasattr(net, "get_cross_attention_stats"):
        info["cross_attention_stats"] = net.get_cross_attention_stats()
    if hasattr(net, "get_cross_attention_grad_stats"):
        info["cross_attention_grad_stats"] = net.get_cross_attention_grad_stats()
    if condition_mode in ("residual_branch", "spatial_residual_branch_v2", "cross_attention_v2"):
        info["residual_shape"] = tuple(condition_residual.shape)
    return info


def _unlock_deform_output_for_render_smoke(gaussians, std):
    if std <= 0:
        return False
    net = gaussians.deform_net.module if hasattr(gaussians.deform_net, "module") else gaussians.deform_net
    final_layer = getattr(getattr(net, "model", None), "up", None)
    if final_layer is None or len(final_layer) < 2:
        return False
    with torch.no_grad():
        torch.nn.init.normal_(final_layer[1].weight, mean=0.0, std=std)
        if final_layer[1].bias is not None:
            torch.nn.init.zeros_(final_layer[1].bias)
    return True


def run_v4_condition_preprocessing_check(device):
    from gaussianavatars.scene.cap4d_gaussian_model_xnemo import CAP4DGaussianModel

    holder = CAP4DGaussianModel.__new__(CAP4DGaussianModel)
    holder.motion_condition_mode = "cross_attention_v4"
    holder.motion_cross_attention_centering = "training_mean"
    holder.motion_cross_attention_mismatch_candidates = 3
    holder.motion_cross_attention_mismatch_selection = "flame_nearest"
    holder._is_training_mode = False
    holder.motion_training_count = None
    holder.timestep = 2

    base = torch.linspace(-1.0, 1.0, 512, device=device)[None]
    offsets = torch.tensor(
        [[0.0], [0.1], [0.2], [0.25], [-0.4]],
        device=device,
    )
    holder.motion_features = base + offsets
    holder.motion_feature_center = holder.motion_features.mean(dim=0, keepdim=True).cpu()
    permuted_center = holder.motion_features.flip(0).mean(dim=0, keepdim=True)
    center_permutation_diff = (
        permuted_center - holder.motion_feature_center.to(device=device)
    ).abs().max()
    if center_permutation_diff.item() > 1e-6:
        raise RuntimeError(
            "cross_attention_v4 center changed under a fixed frame permutation."
        )
    holder.flame_param = {
        "expr": torch.tensor(
            [[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.41, 0.0], [2.0, 0.0]],
            device=device,
        ),
        "eye_rot": torch.zeros(5, 3, device=device),
    }

    zero = torch.zeros(1, 512, device=device)
    zero_condition = holder._condition_motion_feature(zero)
    if torch.count_nonzero(zero_condition).item() != 0:
        raise RuntimeError("cross_attention_v4 centering broke exact zero512 semantics.")

    centered = holder._condition_motion_feature(holder.motion_features)
    if centered.mean(dim=0).abs().max().item() > 1e-6:
        raise RuntimeError("cross_attention_v4 training-mean centering is not zero mean.")

    current = holder.motion_features[[holder.timestep]]
    mismatch = holder._sample_mismatched_motion_feature(
        current,
        device,
        holder.motion_features.dtype,
    )
    if mismatch.shape != current.shape:
        raise RuntimeError(
            "cross_attention_v4 mismatch feature shape changed: "
            f"current={tuple(current.shape)}, mismatch={tuple(mismatch.shape)}"
        )
    if holder._last_motion_mismatch_index == holder.timestep:
        raise RuntimeError("cross_attention_v4 selected the aligned frame as its mismatch.")
    assert_finite("cross_attention_v4_centered_feature", centered)
    assert_finite("cross_attention_v4_mismatch_feature", mismatch)
    return {
        "zero_condition_nonzero": int(torch.count_nonzero(zero_condition).item()),
        "centered_mean_max_abs": float(centered.mean(dim=0).abs().max().cpu()),
        "center_permutation_diff_max": float(center_permutation_diff.cpu()),
        "mismatch_index": int(holder._last_motion_mismatch_index),
        "mismatch_flame_distance": float(holder._last_motion_mismatch_flame_distance),
        "mismatch_condition_cosine": float(holder._last_motion_mismatch_condition_cosine),
    }


def run_render_check(args):
    if not args.render_check:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("--render_check requires CUDA because the Gaussian renderer is CUDA-only.")
    if args.render_config_path is None:
        raise RuntimeError("--render_config_path is required with --render_check.")
    if not args.render_source_paths:
        raise RuntimeError("--render_source_paths is required with --render_check.")
    if args.render_motion_feature_path is None:
        raise RuntimeError("--render_motion_feature_path is required with --render_check.")

    from omegaconf import OmegaConf

    from gaussianavatars.gaussian_renderer.gsplat_renderer import render
    from gaussianavatars.scene.cap4d_gaussian_model_xnemo import CAP4DGaussianModel
    from gaussianavatars.scene.scene import Scene
    from gaussianavatars.train_xnemo import validate_motion_feature_index
    from gaussianavatars.utils.loss_utils import l1_loss

    torch.cuda.set_device(args.render_cuda_device)
    validate_motion_feature_index(args.render_source_paths, args.render_motion_feature_path)
    config = OmegaConf.load(args.render_config_path)
    model_params = config["model_params"]
    opt_params = config["opt_params"]
    model_params["use_motion_condition"] = True
    model_params["motion_feature_path"] = args.render_motion_feature_path
    model_params["motion_condition_mode"] = args.render_condition_mode
    model_params["motion_condition_layers"] = "cross_attention"
    model_params["motion_condition_hidden_dim"] = args.condition_hidden_dim
    model_params["motion_condition_gamma_scale"] = args.condition_gamma_scale
    model_params["motion_condition_norm"] = args.condition_norm
    model_params["motion_condition_gate_init"] = args.condition_gate_init
    model_params["motion_condition_residual_alpha_init"] = args.condition_residual_alpha_init
    model_params["motion_condition_tokens"] = args.condition_tokens
    model_params["motion_condition_attention_dim"] = args.condition_attention_dim
    model_params["motion_cross_attention_gate_init"] = args.cross_attention_gate_init
    model_params["motion_cross_attention_output_init_std"] = args.cross_attention_output_init_std
    model_params["motion_cross_attention_logit_scale"] = args.cross_attention_logit_scale
    model_params["motion_cross_attention_lr_mult"] = args.cross_attention_lr_mult
    model_params["motion_cross_attention_w_decay"] = args.cross_attention_w_decay
    model_params["motion_cross_attention_uv_dropout_prob"] = args.cross_attention_uv_dropout_prob
    model_params["motion_cross_attention_uv_dropout_scale"] = args.cross_attention_uv_dropout_scale
    model_params["motion_cross_attention_uv_noise_std"] = args.cross_attention_uv_noise_std
    model_params["motion_cross_attention_adapter_only"] = args.render_adapter_only
    model_params["motion_cross_attention_centering"] = "training_mean"
    model_params["motion_cross_attention_mismatch_enabled"] = (
        args.render_condition_mode == "cross_attention_v4"
    )
    model_params["motion_cross_attention_mismatch_candidates"] = 4
    model_params["motion_cross_attention_mismatch_selection"] = "flame_nearest"
    model_params["motion_condition_runtime_shuffle"] = args.render_runtime_shuffle
    model_params["motion_nodeform_condition"] = args.render_nodeform_condition
    if args.render_n_gaussians_init > 0:
        model_params["n_gaussians_init"] = args.render_n_gaussians_init

    torch.manual_seed(args.seed)
    model_path = Path(args.render_model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    gaussians = CAP4DGaussianModel(model_params)
    scene = Scene(
        model_path=str(model_path),
        source_paths=args.render_source_paths,
        gaussians=gaussians,
        shuffle=False,
    )
    migrated_checkpoint_keys = 0
    init_checkpoint_iteration = None
    if args.render_init_checkpoint_path is not None:
        checkpoint_weights, init_checkpoint_iteration = torch.load(
            args.render_init_checkpoint_path,
            map_location="cuda",
            weights_only=False,
        )
        gaussians.restore(checkpoint_weights, training_args=None)
        current_state = gaussians.deform_net.state_dict()
        for source_key, source_value in checkpoint_weights["deform_net"].items():
            target_key = gaussians._legacy_unet_key_to_conditional(source_key)
            if target_key not in current_state or current_state[target_key].shape != source_value.shape:
                continue
            if not torch.equal(
                current_state[target_key].detach().cpu(),
                source_value.detach().cpu(),
            ):
                raise RuntimeError(
                    "Migrated baseline U-Net parameter differs from its checkpoint value: "
                    f"source={source_key}, target={target_key}"
                )
            migrated_checkpoint_keys += 1
        if migrated_checkpoint_keys == 0:
            raise RuntimeError("Render smoke did not migrate any baseline U-Net checkpoint keys.")
    gaussians.training_setup(opt_params)
    unlocked_final_layer = _unlock_deform_output_for_render_smoke(
        gaussians,
        args.render_deform_output_probe_std,
    )

    train_cameras = scene.getTrainCameras()
    if len(train_cameras) == 0:
        raise RuntimeError("Render smoke found no training cameras.")
    camera_index = min(max(args.render_camera_index, 0), len(train_cameras) - 1)
    camera = train_cameras[camera_index]
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    gaussians.train()
    gaussians.clear_motion_feature_override()
    gaussians.select_mesh_by_timestep(camera.timestep)
    render_pkg = render(camera, gaussians, background)
    image = render_pkg["render"]
    assert_finite("render_image", image)
    gt_image = camera.original_image.cuda()
    mask = camera.mask[..., None].cuda().float().permute(2, 0, 1)
    loss = l1_loss(image * mask, gt_image * mask)
    loss = loss + gaussians.compute_laplacian_loss() * float(opt_params.get("lambda_laplacian", 0.0))
    actual_condition_residual = None
    actual_residual_ratio = None
    actual_nodeform_residual_max = None
    ratio_penalty = None
    mismatch_penalty = None
    if args.render_condition_mode in ("cross_attention_v3", "cross_attention_v4"):
        actual_condition_residual = gaussians._get_condition_residual()
        if actual_condition_residual is None:
            raise RuntimeError(
                f"{args.render_condition_mode} render smoke did not expose its actual output residual."
            )
        if actual_condition_residual.shape != gaussians.deform_output.shape:
            raise RuntimeError(
                f"{args.render_condition_mode} actual residual shape mismatch: "
                f"residual={tuple(actual_condition_residual.shape)}, "
                f"deform={tuple(gaussians.deform_output.shape)}"
            )
        assert_finite(
            f"{args.render_condition_mode}_actual_residual",
            actual_condition_residual,
        )
        ratio_penalty, actual_residual_ratio = gaussians.compute_motion_residual_ratio_loss(
            args.render_residual_ratio_limit
        )
        nodeform_residual = (
            gaussians.cross_attention_v3_nodeform_output
            if args.render_condition_mode == "cross_attention_v3"
            else gaussians.cross_attention_v4_nodeform_output
        )
        actual_nodeform_residual_max = float(
            nodeform_residual.detach().abs().max().cpu()
        )
        assert_finite(f"{args.render_condition_mode}_ratio_penalty", ratio_penalty)
        loss = loss + ratio_penalty * args.render_residual_ratio_loss_weight
        if args.render_condition_mode == "cross_attention_v4":
            mismatch_penalty = gaussians.compute_motion_mismatch_suppression_loss()
            assert_finite("cross_attention_v4_mismatch_penalty", mismatch_penalty)
            if gaussians.cross_attention_v4_mismatch_deform_output is None:
                raise RuntimeError(
                    "cross_attention_v4 render smoke did not expose its mismatched-512 residual."
                )
            loss = loss + mismatch_penalty * args.render_mismatch_loss_weight
    loss.backward()

    stats = gaussians.collect_condition_monitor_stats(include_sensitivity=False)
    cross_attention_grad_max = max(
        [
            float(value)
            for key, value in stats.items()
            if key.startswith("condition/cross_attention/") and key.endswith("_grad_norm")
        ]
        or [0.0]
    )
    if cross_attention_grad_max <= args.min_render_condition_grad:
        raise RuntimeError(
            "Render smoke produced too-small cross-attention gradients: "
            f"max_grad={cross_attention_grad_max:.6e}"
        )

    fixed_uv_deform_diff = 0.0
    render_condition_diff = 0.0
    if gaussians.motion_features is not None and gaussians.motion_features.shape[0] > 1:
        timestep = int(camera.timestep)
        current_feature = gaussians.motion_features[[timestep]].cuda()
        alt_index = (timestep + 1) % gaussians.motion_features.shape[0]
        alt_feature = gaussians.motion_features[[alt_index]].cuda()
        gaussians.eval()
        with torch.no_grad():
            gaussians.select_mesh_by_timestep(camera.timestep)
            fixed_uv_offsets = gaussians._last_unet_uv_offsets.detach()
            current_deform, _ = gaussians.forward_unet(
                fixed_uv_offsets,
                motion_feature_override=current_feature,
                save_debug=False,
            )
            alt_deform, _ = gaussians.forward_unet(
                fixed_uv_offsets,
                motion_feature_override=alt_feature,
                save_debug=False,
            )
            fixed_uv_deform_diff = (current_deform - alt_deform).abs().max().item()
            gaussians.set_motion_feature_override(current_feature)
            gaussians.select_mesh_by_timestep(camera.timestep)
            current_render = render(camera, gaussians, background)["render"]
            gaussians.set_motion_feature_override(alt_feature)
            gaussians.select_mesh_by_timestep(camera.timestep)
            alt_render = render(camera, gaussians, background)["render"]
            render_condition_diff = (current_render - alt_render).abs().max().item()
            gaussians.clear_motion_feature_override()
            gaussians.select_mesh_by_timestep(camera.timestep)
    if fixed_uv_deform_diff <= args.min_render_condition_diff:
        raise RuntimeError(
            "Render smoke fixed-UV deformation did not change under a different 512 condition: "
            f"max_diff={fixed_uv_deform_diff:.6e}"
        )

    result = {
        "render_image_shape": tuple(image.shape),
        "render_loss": float(loss.detach().cpu()),
        "render_total_grad_norm": grad_norm(gaussians.deform_net.parameters()),
        "render_cross_attention_grad_max": cross_attention_grad_max,
        "render_fixed_uv_deform_condition_diff_max": fixed_uv_deform_diff,
        "render_image_condition_diff_max": render_condition_diff,
        "render_unlocked_final_layer_for_smoke": unlocked_final_layer,
        "render_init_checkpoint_iteration": init_checkpoint_iteration,
        "render_migrated_checkpoint_keys": migrated_checkpoint_keys,
        "render_condition_stats": stats,
    }
    if actual_condition_residual is not None:
        result.update(
            {
                "render_actual_condition_residual_shape": tuple(actual_condition_residual.shape),
                "render_actual_condition_residual_mean_abs": float(
                    actual_condition_residual.detach().abs().mean().cpu()
                ),
                "render_actual_condition_residual_ratio": float(
                    actual_residual_ratio.detach().cpu()
                ),
                "render_actual_condition_residual_ratio_penalty": float(
                    ratio_penalty.detach().cpu()
                ),
                "render_nodeform_condition_residual_max_abs": float(
                    actual_nodeform_residual_max
                ),
            }
        )
    if mismatch_penalty is not None:
        result["render_mismatch_suppression_penalty"] = float(
            mismatch_penalty.detach().cpu()
        )
        captured_center = gaussians.capture().get("motion_feature_center")
        if captured_center is None or tuple(captured_center.shape) != (1, 512):
            raise RuntimeError(
                "cross_attention_v4 checkpoint capture omitted the [1,512] training center."
            )
        result["render_checkpoint_center_shape"] = tuple(captured_center.shape)
    return result


def run_expression_balance_check():
    from gaussianavatars.train_xnemo import (
        build_source_sampling_weights,
        compute_region_loss_terms,
        make_camera_loader,
        source_value_for_camera,
    )
    from gaussianavatars.utils.region_loss_utils import RegionMaskProjector

    class DummyCamera:
        def __init__(self, source_id):
            self.source_id = source_id
            self.is_pseudo = False
            self.image_name = None

    cameras = [DummyCamera(0)] + [DummyCamera(1) for _ in range(9)] + [
        DummyCamera(2) for _ in range(2)
    ]
    target_probabilities = [0.05, 0.45, 0.50]
    weights, probabilities, counts = build_source_sampling_weights(
        cameras,
        target_probabilities,
    )
    source_weight_mass = []
    for source_id in range(len(target_probabilities)):
        source_weight_mass.append(
            float(
                weights[
                    torch.tensor([camera.source_id == source_id for camera in cameras])
                ].sum()
            )
        )
    if not torch.allclose(
        torch.tensor(source_weight_mass, dtype=torch.float64),
        probabilities,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Source-balanced weights do not preserve target source probabilities: "
            f"mass={source_weight_mass}, target={probabilities.tolist()}"
        )

    source_regularizer_scales = [1.0, 1.0, 0.25]
    selected_scale = source_value_for_camera(
        cameras[-1],
        source_regularizer_scales,
    )
    if selected_scale != 0.25:
        raise RuntimeError(
            "Per-source regularizer lookup selected the wrong source scale: "
            f"got={selected_scale}"
        )

    loader, _, _ = make_camera_loader(
        cameras,
        target_probabilities,
        seed=1234,
        num_workers=0,
    )
    sampled_camera = next(iter(loader))
    if sampled_camera.source_id not in (0, 1, 2):
        raise RuntimeError("Weighted camera DataLoader returned an invalid source.")

    cheek_vertices = torch.tensor(
        [
            [-0.050, -0.090, 0.020],
            [-0.070, -0.045, 0.030],
            [-0.090, 0.000, 0.020],
            [0.050, -0.090, 0.020],
            [0.070, -0.045, 0.030],
            [0.090, 0.000, 0.020],
        ],
        dtype=torch.float32,
    )
    projector = RegionMaskProjector(cheek_vertices)
    if projector.region_vertex_counts.get("cheeks") != len(cheek_vertices):
        raise RuntimeError(
            "Cheek region projector did not retain both cheek components: "
            f"counts={projector.region_vertex_counts}"
        )

    region_pred = torch.zeros(3, 4, 4)
    region_target = torch.ones_like(region_pred)
    region_masks = {
        name: torch.ones(1, 4, 4)
        for name in ("mouth", "eyes", "brow", "cheeks")
    }
    historical_region_terms, _ = compute_region_loss_terms(
        region_pred,
        region_target,
        region_masks,
        {"region_loss_type": "l1"},
    )
    if float(historical_region_terms["region_cheeks"]) != 0.0:
        raise RuntimeError("Cheek supervision must be opt-in for historical reproducibility.")

    return {
        "target_probabilities": probabilities.tolist(),
        "train_source_counts": counts.tolist(),
        "source_weight_mass": source_weight_mass,
        "action_regularizer_scale": selected_scale,
        "region_vertex_counts": projector.region_vertex_counts,
        "historical_cheek_loss_weighted": float(historical_region_terms["region_cheeks"]),
    }


def run(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    strict_norm_stats = verify_strict_normalizers(device)
    modes = args.condition_modes
    if "all" in modes:
        modes = [
            "strict_adain_allnorm",
            "spatial_residual_branch_v2",
            "cross_attention",
            "cross_attention_v2",
            "cross_attention_v3",
            "cross_attention_v4",
        ]

    baseline_info = run_baseline_check(args, device)
    legacy_info = run_legacy_concat_check(args, device)
    v4_preprocessing_info = run_v4_condition_preprocessing_check(device)
    print("baseline:", baseline_info)
    print("legacy_concat:", legacy_info)
    print("cross_attention_v4_preprocessing:", v4_preprocessing_info)
    for mode in modes:
        info = run_conditional_mode(args, device, mode)
        print("condition_mode:", mode)
        for key, value in info.items():
            print(f"{mode}/{key}:", value)
    render_info = run_render_check(args)
    if render_info is not None:
        print(f"render_check: {args.render_condition_mode}")
        for key, value in render_info.items():
            print(f"render/{key}:", value)
    expression_balance_info = run_expression_balance_check()
    print("expression_balance:", expression_balance_info)
    print("strict_normalizer_zero_affine_max_errors:", tuple(f"{v:.3e}" for v in strict_norm_stats))
    print("smoke_test: ok")


if __name__ == "__main__":
    parser = ArgumentParser(description="Smoke-test Xnemo legacy concat and conditional-norm U-Net paths.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--uv_resolution", type=int, default=128)
    parser.add_argument("--n_unet_layers", type=int, default=6)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--pos_encoding_channels", type=int, default=24)
    parser.add_argument("--motion_condition_channels", type=int, default=16)
    parser.add_argument("--condition_hidden_dim", type=int, default=128)
    parser.add_argument(
        "--condition_modes",
        nargs="+",
        choices=[
            "all",
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
        default=[
            "strict_adain_allnorm",
            "spatial_residual_branch_v2",
            "cross_attention",
            "cross_attention_v2",
            "cross_attention_v3",
            "cross_attention_v4",
        ],
    )
    parser.add_argument("--condition_layers", choices=["bottleneck", "gated_multistage", "residual_branch", "allnorm", "spatial_residual_v2", "cross_attention"], default=None)
    parser.add_argument("--condition_gamma_scale", type=float, default=0.1)
    parser.add_argument("--condition_norm", choices=["existing", "group", "instance"], default="existing")
    parser.add_argument("--condition_gate_init", type=float, default=0.05)
    parser.add_argument("--condition_residual_alpha_init", type=float, default=0.05)
    parser.add_argument("--condition_tokens", type=int, default=8)
    parser.add_argument("--condition_attention_dim", type=int, default=64)
    parser.add_argument("--cross_attention_gate_init", type=float, default=0.2)
    parser.add_argument("--cross_attention_output_init_std", type=float, default=2e-2)
    parser.add_argument("--cross_attention_logit_scale", type=float, default=4.0)
    parser.add_argument("--cross_attention_lr_mult", type=float, default=10.0)
    parser.add_argument("--cross_attention_w_decay", type=float, default=0.0)
    parser.add_argument("--cross_attention_uv_dropout_prob", type=float, default=0.0)
    parser.add_argument("--cross_attention_uv_dropout_scale", type=float, default=0.0)
    parser.add_argument("--cross_attention_uv_noise_std", type=float, default=0.0)
    parser.add_argument("--condition_probe_std", type=float, default=1e-3)
    parser.add_argument("--min_condition_diff", type=float, default=1e-8)
    parser.add_argument("--min_spatial_diff", type=float, default=1e-8)
    parser.add_argument("--render_check", action="store_true", default=False)
    parser.add_argument("--render_config_path", type=str, default=None)
    parser.add_argument("--render_source_paths", nargs="+", default=None)
    parser.add_argument("--render_motion_feature_path", type=str, default=None)
    parser.add_argument("--render_init_checkpoint_path", type=str, default=None)
    parser.add_argument("--render_adapter_only", action="store_true", default=False)
    parser.add_argument("--render_model_path", type=str, default="/tmp/cap4d_xnemo_cross_attention_smoke")
    parser.add_argument("--render_camera_index", type=int, default=0)
    parser.add_argument("--render_cuda_device", type=int, default=0)
    parser.add_argument("--render_runtime_shuffle", choices=["none", "batch"], default="none")
    parser.add_argument(
        "--render_condition_mode",
        choices=[
            "cross_attention",
            "cross_attention_v2",
            "cross_attention_v3",
            "cross_attention_v4",
        ],
        default="cross_attention_v4",
    )
    parser.add_argument("--render_nodeform_condition", choices=["zero", "mean", "neutral"], default="zero")
    parser.add_argument("--render_n_gaussians_init", type=int, default=-1)
    parser.add_argument("--render_deform_output_probe_std", type=float, default=1e-4)
    parser.add_argument("--render_residual_ratio_limit", type=float, default=0.35)
    parser.add_argument("--render_residual_ratio_loss_weight", type=float, default=0.02)
    parser.add_argument("--render_mismatch_loss_weight", type=float, default=0.05)
    parser.add_argument("--min_render_condition_grad", type=float, default=1e-12)
    parser.add_argument("--min_render_condition_diff", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    run(parser.parse_args())
