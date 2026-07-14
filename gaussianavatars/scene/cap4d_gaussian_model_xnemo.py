# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import roma
from PIL import Image

from cap4d.flame.flame import CAP4DFlameSkinner
from cap4d.mmdm.conditioning.mesh2img import VertexShader

from gaussianavatars.scene.net.positional_encoding import get_pos_enc
from gaussianavatars.scene.net.unet import define_G
from gaussianavatars.scene.gaussian_model import GaussianModel
from gaussianavatars.utils.obj_io import load_obj
from gaussianavatars.utils.mesh_utils import gen_uv_mesh
from gaussianavatars.utils.graphics_utils import compute_face_orientation
from gaussianavatars.utils.general_utils import get_expon_lr_func
from gaussianavatars.utils.sh_utils import RGB2SH


FLAME_TEMPLATE_PATH = "data/assets/flame/cap4d_avatar_template.obj"
BACK_HEAD_FACE_IDS_PATH = "data/assets/flame/back_head_region/back_head_face_ids.npy"
BACK_HEAD_UV_TRANSITION_PATH = "data/assets/flame/back_head_region/back_head_uv_transition.png"
STD_DEFORM = 0.0108


class CAP4DGaussianModel(GaussianModel):
    def __init__(
        self, 
        model_params: Dict,
    ):
        super().__init__(model_params["sh_degree"])
        
        # Load FLAME skinner
        self.flame_model = CAP4DFlameSkinner(
            n_shape_params=150,
            n_expr_params=65,
            add_mouth=True,
            add_lower_jaw=model_params["use_lower_jaw"],
        ).cuda()

        # Load template mesh
        flame_verts, flame_faces, flame_aux = load_obj(FLAME_TEMPLATE_PATH)

        self.flame_verts = flame_verts.cuda()
        self.flame_faces_uvs = flame_faces.textures_idx.cuda()
        self.flame_faces = flame_faces.verts_idx.cuda()
        self.flame_uvs = flame_aux.verts_uvs.cuda()
        self.flame_uvs = self.flame_uvs * 2. - 1.
        self.flame_uvs[..., 1] = -self.flame_uvs[..., 1]

        self.flame_param = None
        self.static_neck = model_params["static_neck"]
        self.gaussian_init_type = model_params["gaussian_init_type"]
        self.n_gaussians_init = model_params["n_gaussians_init"]
        self.enable_back_head_densify = model_params.get("enable_back_head_densify", False)
        self.back_head_face_ids_path = model_params.get("back_head_face_ids_path", BACK_HEAD_FACE_IDS_PATH)
        self.back_head_density_multiplier = model_params.get("back_head_density_multiplier", 3)
        self.back_head_extra_opacity = model_params.get("back_head_extra_opacity", 0.35)
        self.back_head_extra_scale_mult = model_params.get("back_head_extra_scale_mult", 0.75)
        self.hair_color_rgb = self._load_hair_color(model_params) if self.enable_back_head_densify else None

        self.uv_resolution = model_params["uv_resolution"]
        self.n_points_per_triangle = model_params["n_points_per_triangle"]
        self.use_expr_mask = model_params["use_expr_mask"]
        self.use_motion_condition = bool(model_params.get("use_motion_condition", False))
        self.motion_feature_path = model_params.get("motion_feature_path", None)
        self.motion_feature_align = model_params.get("motion_feature_align", "strict")
        self.motion_condition_mode = str(model_params.get("motion_condition_mode", "legacy_concat")).lower()
        if self.motion_condition_mode == "adain":
            print("WARNING: motion_condition_mode='adain' is treated as strict_adain.")
            self.motion_condition_mode = "strict_adain"
        if self.motion_condition_mode == "strict_adain_bottleneck":
            self.motion_condition_mode = "strict_adain"
        self.motion_condition_channels = int(model_params.get("motion_condition_channels", 16))
        self.motion_condition_layers = model_params.get("motion_condition_layers", "bottleneck")
        if self.motion_condition_mode == "gated_multistage" and self.motion_condition_layers == "bottleneck":
            self.motion_condition_layers = "gated_multistage"
        if self.motion_condition_mode == "residual_branch" and self.motion_condition_layers == "bottleneck":
            self.motion_condition_layers = "residual_branch"
        if self.motion_condition_mode == "strict_adain_allnorm":
            self.motion_condition_layers = "allnorm"
        if self.motion_condition_mode == "spatial_residual_branch_v2":
            self.motion_condition_layers = "spatial_residual_v2"
        if self.motion_condition_mode in (
            "cross_attention",
            "cross_attention_v2",
            "cross_attention_v3",
            "cross_attention_v4",
        ):
            self.motion_condition_layers = "cross_attention"
        self.motion_condition_hidden_dim = int(model_params.get("motion_condition_hidden_dim", 128))
        self.motion_condition_gamma_scale = float(model_params.get("motion_condition_gamma_scale", 0.1))
        self.motion_condition_norm = model_params.get("motion_condition_norm", "existing")
        self.motion_condition_gate_init = float(model_params.get("motion_condition_gate_init", 0.05))
        self.motion_condition_residual_alpha_init = float(model_params.get("motion_condition_residual_alpha_init", 0.05))
        self.motion_condition_tokens = int(model_params.get("motion_condition_tokens", 8))
        self.motion_condition_attention_dim = int(model_params.get("motion_condition_attention_dim", 64))
        self.motion_cross_attention_gate_init = float(
            model_params.get("motion_cross_attention_gate_init", 0.2)
        )
        self.motion_cross_attention_output_init_std = float(
            model_params.get("motion_cross_attention_output_init_std", 2e-2)
        )
        self.motion_cross_attention_logit_scale = float(
            model_params.get("motion_cross_attention_logit_scale", 4.0)
        )
        self.motion_cross_attention_lr_mult = float(
            model_params.get("motion_cross_attention_lr_mult", 10.0)
        )
        self.motion_cross_attention_w_decay = float(
            model_params.get("motion_cross_attention_w_decay", 0.0)
        )
        self.motion_cross_attention_uv_dropout_prob = float(
            model_params.get("motion_cross_attention_uv_dropout_prob", 0.0)
        )
        self.motion_cross_attention_uv_dropout_scale = float(
            model_params.get("motion_cross_attention_uv_dropout_scale", 0.0)
        )
        self.motion_cross_attention_uv_noise_std = float(
            model_params.get("motion_cross_attention_uv_noise_std", 0.0)
        )
        self.motion_cross_attention_base_pretrain_iters = int(
            model_params.get("motion_cross_attention_base_pretrain_iters", 0)
        )
        self.motion_cross_attention_base_lr_mult_after_pretrain = float(
            model_params.get("motion_cross_attention_base_lr_mult_after_pretrain", 1.0)
        )
        self.motion_cross_attention_condition_warmup_iters = int(
            model_params.get("motion_cross_attention_condition_warmup_iters", 0)
        )
        self.motion_cross_attention_adapter_only = bool(
            model_params.get("motion_cross_attention_adapter_only", False)
        )
        self.motion_cross_attention_centering = str(
            model_params.get("motion_cross_attention_centering", "training_mean")
        ).lower()
        self.motion_cross_attention_mismatch_enabled = bool(
            model_params.get("motion_cross_attention_mismatch_enabled", False)
        )
        self.motion_cross_attention_mismatch_candidates = int(
            model_params.get("motion_cross_attention_mismatch_candidates", 16)
        )
        self.motion_cross_attention_mismatch_selection = str(
            model_params.get(
                "motion_cross_attention_mismatch_selection",
                "flame_nearest",
            )
        ).lower()
        if not 0.0 <= self.motion_cross_attention_uv_dropout_prob <= 1.0:
            raise ValueError("motion_cross_attention_uv_dropout_prob must be in [0, 1].")
        if self.motion_cross_attention_uv_dropout_scale < 0.0:
            raise ValueError("motion_cross_attention_uv_dropout_scale must be non-negative.")
        if self.motion_cross_attention_uv_noise_std < 0.0:
            raise ValueError("motion_cross_attention_uv_noise_std must be non-negative.")
        if self.motion_cross_attention_base_pretrain_iters < 0:
            raise ValueError("motion_cross_attention_base_pretrain_iters must be non-negative.")
        if self.motion_cross_attention_base_lr_mult_after_pretrain < 0.0:
            raise ValueError("motion_cross_attention_base_lr_mult_after_pretrain must be non-negative.")
        if self.motion_cross_attention_condition_warmup_iters < 0:
            raise ValueError("motion_cross_attention_condition_warmup_iters must be non-negative.")
        if self.motion_cross_attention_centering not in ("none", "training_mean"):
            raise ValueError(
                "motion_cross_attention_centering must be 'none' or 'training_mean'."
            )
        if self.motion_cross_attention_mismatch_candidates < 1:
            raise ValueError(
                "motion_cross_attention_mismatch_candidates must be at least 1."
            )
        if self.motion_cross_attention_mismatch_selection not in (
            "random",
            "flame_nearest",
        ):
            raise ValueError(
                "motion_cross_attention_mismatch_selection must be 'random' or "
                "'flame_nearest'."
            )
        if (
            self.motion_cross_attention_adapter_only
            and self.motion_condition_mode not in (
                "cross_attention_v2",
                "cross_attention_v3",
                "cross_attention_v4",
            )
        ):
            raise ValueError(
                "motion_cross_attention_adapter_only requires cross_attention_v2, "
                "cross_attention_v3, or cross_attention_v4."
            )
        if (
            self.motion_cross_attention_mismatch_enabled
            and self.motion_condition_mode != "cross_attention_v4"
        ):
            raise ValueError(
                "motion_cross_attention_mismatch_enabled requires cross_attention_v4."
            )
        self.motion_condition_runtime_shuffle = str(
            model_params.get("motion_condition_runtime_shuffle", "none")
        ).lower()
        if self.motion_condition_runtime_shuffle not in ("none", "batch"):
            raise ValueError(
                f"Unsupported motion_condition_runtime_shuffle={self.motion_condition_runtime_shuffle!r}. "
                "Use 'none' or 'batch'."
            )
        self.motion_nodeform_condition = str(
            model_params.get("motion_nodeform_condition", "zero")
        ).lower()
        if self.motion_nodeform_condition not in ("zero", "mean", "neutral"):
            raise ValueError(
                f"Unsupported motion_nodeform_condition={self.motion_nodeform_condition!r}. "
                "Use 'zero', 'mean', or 'neutral'."
            )
        self.motion_neutral_index = None
        self._motion_feature_override = None
        self._is_training_mode = False
        self.current_flame_verts_for_region_masks = None
        self.use_legacy_motion_concat = self.use_motion_condition and self.motion_condition_mode == "legacy_concat"
        self.use_conditional_norm = self.use_motion_condition and self.motion_condition_mode in (
            "strict_adain",
            "strict_adain_allnorm",
            "conditional_norm",
            "film",
            "gated_multistage",
            "residual_branch",
            "spatial_residual_branch_v2",
            "cross_attention",
            "cross_attention_v2",
            "cross_attention_v3",
            "cross_attention_v4",
        )
        if self.use_motion_condition and not (self.use_legacy_motion_concat or self.use_conditional_norm):
            raise ValueError(
                f"Unsupported motion_condition_mode={self.motion_condition_mode!r}. "
                "Use 'legacy_concat', 'strict_adain', 'strict_adain_allnorm', "
                "'film', 'gated_multistage', 'residual_branch', or "
                "'spatial_residual_branch_v2', 'cross_attention', 'cross_attention_v2', "
                "'cross_attention_v3', or 'cross_attention_v4'."
            )
        self.motion_features = None
        self.motion_training_count = None
        self.motion_feature_center = torch.zeros((1, 512), dtype=torch.float32)
        self.motion_feature_center_source = "zeros"
        self.motion_feature_common_energy_ratio = 0.0
        self.motion_feature_centered_rms = 0.0
        self._motion_feature_center_restored = False
        self.spatial_residual_v2_base_actual_full = None
        self.spatial_residual_v2_delta_actual_full = None
        self.spatial_residual_v2_deform_output = None
        self.spatial_residual_v2_nodeform_output = None
        self.cross_attention_v3_base_actual_full = None
        self.cross_attention_v3_delta_actual_full = None
        self.cross_attention_v3_deform_output = None
        self.cross_attention_v3_nodeform_output = None
        self.cross_attention_v4_base_actual_full = None
        self.cross_attention_v4_delta_actual_full = None
        self.cross_attention_v4_mismatch_delta_actual_full = None
        self.cross_attention_v4_deform_output = None
        self.cross_attention_v4_nodeform_output = None
        self.cross_attention_v4_mismatch_deform_output = None
        self._last_motion_mismatch_index = None
        self._last_motion_mismatch_flame_distance = 0.0
        self._last_motion_mismatch_condition_cosine = 0.0
        self.deform_base_output = None
        self.neutral_base_output = None
        self._last_cross_attention_uv_dropout_fraction = 0.0
        self._last_cross_attention_uv_noise_std = 0.0
        self._cross_attention_base_pretrain_active = False
        self._cross_attention_condition_lr_scale = 1.0
        self._cross_attention_base_lr_scale = 1.0
        self.enable_back_static_mask = model_params.get("enable_back_static_mask", False)
        self.back_static_mask_path = model_params.get("back_static_mask_path", BACK_HEAD_UV_TRANSITION_PATH)
        self.back_static_mode = model_params.get("back_static_mode", "neutral")
        self.save_deform_debug = model_params.get("save_deform_debug", False)
        self.deform_debug_dir = Path(model_params.get("deform_debug_dir", "outputs/deform_debug"))
        self.deform_debug_interval = model_params.get("deform_debug_interval", 100)
        self._deform_debug_counter = 0
        self.back_static_mask = None
        self._load_back_static_mask()

        n_pos_enc = 12
        self.pos_enc = get_pos_enc(n_pos_enc, self.uv_resolution).cuda()

        deform_in_channels = 3 + n_pos_enc * 2
        if self.use_legacy_motion_concat:
            deform_in_channels += self.motion_condition_channels
            self.motion_condition_proj = nn.Sequential(
                nn.LayerNorm(512),
                nn.Linear(512, 64),
                nn.SiLU(),
                nn.Linear(64, self.motion_condition_channels),
            ).cuda()
            print(
                "Enabled Xnemo motion conditioning:",
                f"mode={self.motion_condition_mode}",
                f"channels={self.motion_condition_channels}",
                f"feature_path={self.motion_feature_path}",
                f"align={self.motion_feature_align}",
            )
        else:
            self.motion_condition_proj = None
            if self.use_conditional_norm:
                print(
                    "Enabled Xnemo motion conditioning:",
                    f"mode={self.motion_condition_mode}",
                    f"layers={self.motion_condition_layers}",
                    f"hidden_dim={self.motion_condition_hidden_dim}",
                    f"gamma_scale={self.motion_condition_gamma_scale}",
                    f"film_norm={self.motion_condition_norm}",
                    f"gate_init={self.motion_condition_gate_init}",
                    f"residual_alpha_init={self.motion_condition_residual_alpha_init}",
                    f"tokens={self.motion_condition_tokens}",
                    f"attention_dim={self.motion_condition_attention_dim}",
                    f"cross_attention_gate_init={self.motion_cross_attention_gate_init}",
                    f"cross_attention_output_init_std={self.motion_cross_attention_output_init_std}",
                    f"cross_attention_logit_scale={self.motion_cross_attention_logit_scale}",
                    f"cross_attention_lr_mult={self.motion_cross_attention_lr_mult}",
                    f"cross_attention_uv_dropout_prob={self.motion_cross_attention_uv_dropout_prob}",
                    f"cross_attention_uv_dropout_scale={self.motion_cross_attention_uv_dropout_scale}",
                    f"cross_attention_uv_noise_std={self.motion_cross_attention_uv_noise_std}",
                    f"cross_attention_base_pretrain_iters={self.motion_cross_attention_base_pretrain_iters}",
                    f"cross_attention_base_lr_mult_after_pretrain={self.motion_cross_attention_base_lr_mult_after_pretrain}",
                    f"cross_attention_condition_warmup_iters={self.motion_cross_attention_condition_warmup_iters}",
                    f"cross_attention_adapter_only={self.motion_cross_attention_adapter_only}",
                    f"cross_attention_centering={self.motion_cross_attention_centering}",
                    f"cross_attention_mismatch_enabled={self.motion_cross_attention_mismatch_enabled}",
                    f"cross_attention_mismatch_candidates={self.motion_cross_attention_mismatch_candidates}",
                    f"cross_attention_mismatch_selection={self.motion_cross_attention_mismatch_selection}",
                    f"nodeform_condition={self.motion_nodeform_condition}",
                    f"feature_path={self.motion_feature_path}",
                    f"align={self.motion_feature_align}",
                )

        self.deform_net = define_G(
            deform_in_channels,
            3, 
            64, 
            f'unet_{self.uv_resolution}', 
            n_layers=model_params["n_unet_layers"], 
            norm="instance",
            condition_mode=self.motion_condition_mode if self.use_conditional_norm else "legacy_concat",
            condition_dim=512,
            condition_layers=self.motion_condition_layers,
            condition_hidden_dim=self.motion_condition_hidden_dim,
            condition_gamma_scale=self.motion_condition_gamma_scale,
            condition_norm=self.motion_condition_norm,
            condition_gate_init=self.motion_condition_gate_init,
            condition_residual_alpha_init=self.motion_condition_residual_alpha_init,
            condition_num_tokens=self.motion_condition_tokens,
            condition_attention_dim=self.motion_condition_attention_dim,
            condition_cross_attention_gate_init=self.motion_cross_attention_gate_init,
            condition_attention_output_init_std=self.motion_cross_attention_output_init_std,
            condition_attention_logit_scale=self.motion_cross_attention_logit_scale,
            condition_attention_direct_tokens=self.motion_condition_mode in (
                "cross_attention_v2",
                "cross_attention_v3",
                "cross_attention_v4",
            ),
            condition_attention_use_position=self.motion_condition_mode in (
                "cross_attention_v2",
                "cross_attention_v3",
                "cross_attention_v4",
            ),
        ).cuda()
        if self.motion_condition_mode in (
            "cross_attention",
            "cross_attention_v2",
            "cross_attention_v3",
            "cross_attention_v4",
        ):
            cross_attention_sites = self.deform_net.get_cross_attention_site_table()
            cross_attention_params = sum(row["parameters"] for row in cross_attention_sites)
            print(
                "Cross-attention injection sites:",
                [row["site"] for row in cross_attention_sites],
                f"parameters={cross_attention_params}",
            )
            if self.motion_condition_mode in ("cross_attention_v3", "cross_attention_v4"):
                if len(cross_attention_sites) != 3 or any(
                    "output_cross_attention_c3" in row["site"] for row in cross_attention_sites
                ):
                    raise RuntimeError(
                        f"{self.motion_condition_mode} must inject only at "
                        "c256/c128/c64 decoder features."
                    )
        with torch.no_grad():
            # Initialize final deformation layer with zeros so that initial deformation is zero
            if hasattr(self.deform_net, "zero_last_layer"):
                self.deform_net.zero_last_layer()
            else:
                self.deform_net.model.model[-1].weight.data *= 0
                self.deform_net.model.model[-1].bias.data *= 0

        self.load_uv()

        if self.use_motion_condition and self.motion_feature_path is not None:
            self.load_motion_features(self.motion_feature_path)

    def _set_buffer(self, name: str, tensor: torch.Tensor):
        # GaussianModel is not an nn.Module; keep non-parameter tensors in a buffer-like registry.
        if not hasattr(self, "_buffers"):
            self._buffers = {}
        setattr(self, name, tensor)
        self._buffers[name] = tensor

    def register_buffer(self, name: str, tensor: torch.Tensor):
        self._set_buffer(name, tensor)

    def _load_back_static_mask(self):
        if not self.enable_back_static_mask:
            return

        mask_path = Path(self.back_static_mask_path)
        if not mask_path.exists():
            raise FileNotFoundError(
                f"enable_back_static_mask=True but back static mask was not found: {mask_path}"
            )

        if self.back_static_mode not in ("neutral", "zero"):
            raise ValueError(
                f"Unsupported back_static_mode={self.back_static_mode!r}. Use 'neutral' or 'zero'."
            )

        with Image.open(mask_path) as mask_file:
            mask_image = mask_file.convert("L")
            if mask_image.size != (self.uv_resolution, self.uv_resolution):
                mask_image = mask_image.resize(
                    (self.uv_resolution, self.uv_resolution),
                    resample=Image.BILINEAR,
                )
            mask = np.array(mask_image, dtype=np.float32) / 255.

        mask_tensor = torch.from_numpy(mask)[None, None].clamp(0., 1.).cuda()
        self.register_buffer("back_static_mask", mask_tensor)
        print(
            "Loaded back static deformation mask:",
            mask_path,
            "shape:",
            tuple(mask_tensor.shape),
            "mode:",
            self.back_static_mode,
            "range:",
            (float(mask_tensor.min()), float(mask_tensor.max())),
        )

    def _load_hair_color(self, model_params: Dict):
        color = model_params.get("back_head_hair_color", None)
        if color is not None:
            color = np.array(color, dtype=np.float32)
            if color.max() > 1.:
                color = color / 255.
            print(f"Back-head hair color from config: {color.tolist()}")
            return torch.tensor(color, dtype=torch.float32, device="cuda")

        image_path = model_params.get("back_head_hair_color_image", None)
        if image_path is None:
            image_path = model_params.get("hair_color_image", None)
        if image_path is not None:
            image_path = Path(image_path)
            if image_path.exists():
                with Image.open(image_path) as img_file:
                    img = np.array(img_file.convert("RGB"), dtype=np.float32) / 255.
                brightness = img.mean(axis=-1)
                fg = brightness < 0.95
                if fg.any():
                    pixels = img[fg]
                else:
                    pixels = img.reshape(-1, 3)
                dark_threshold = np.quantile(pixels.mean(axis=-1), 0.25)
                hair_pixels = pixels[pixels.mean(axis=-1) <= dark_threshold]
                color = np.median(hair_pixels, axis=0).astype(np.float32)
                print(f"Back-head hair color estimated from {image_path}: {color.tolist()}")
                return torch.tensor(color, dtype=torch.float32, device="cuda")
            print(f"WARNING: back_head_hair_color_image does not exist: {image_path}")

        color = np.array([0.015, 0.025, 0.04], dtype=np.float32)
        print(f"WARNING: using default dark back-head hair color: {color.tolist()}")
        return torch.tensor(color, dtype=torch.float32, device="cuda")

    def _load_back_head_flame_face_mask(self):
        face_ids_path = Path(self.back_head_face_ids_path)
        face_mask = torch.zeros(self.flame_faces.shape[0], dtype=torch.bool, device="cuda")
        if not face_ids_path.exists():
            print(f"WARNING: back_head_face_ids.npy not found at {face_ids_path}. Back-head densification disabled.")
            return face_mask

        face_ids = np.load(face_ids_path).astype(np.int64)
        face_ids = face_ids[(face_ids >= 0) & (face_ids < self.flame_faces.shape[0])]
        if len(face_ids) == 0:
            print(f"WARNING: no valid back-head face ids found in {face_ids_path}. Back-head densification disabled.")
            return face_mask

        face_mask[torch.tensor(face_ids, dtype=torch.long, device="cuda")] = True
        print(f"Loaded back-head FLAME faces: {int(face_mask.sum())} from {face_ids_path}")
        return face_mask

    @torch.no_grad()
    def load_uv(self):
        self.vert_shader = VertexShader().cuda()

        deformable_vertices = np.genfromtxt("data/assets/flame/deformable_verts.txt").astype(np.int64)
        vert_mask = torch.zeros_like(self.flame_verts[:, 0]).cuda()
        vert_mask[deformable_vertices] = 1
        deformable_face_mask = vert_mask[self.flame_faces]
        deformable_face_mask = deformable_face_mask.min(dim=-1)[0]

        # create pix_to_face map for UV rasterization and remeshing
        shader_input = {
            "positions": torch.cat([self.flame_uvs, torch.ones_like(self.flame_uvs[:, [1]])], dim=-1)[None],
        }
        _, fragments = self.vert_shader(
            shader_input, 
            self.flame_faces_uvs[None], 
            None, 
            None, 
            (self.uv_resolution, self.uv_resolution), 
            0.
        )
        self.fragments = fragments

        pix_to_face = self.fragments.pix_to_face
        uv_mask = pix_to_face >= 0

        self.uv_mask = uv_mask.permute(0, 3, 1, 2)

        pix_to_face[pix_to_face < 0] = 0
        if self.enable_back_head_densify:
            self.back_head_flame_face_mask = self._load_back_head_flame_face_mask()
        else:
            self.back_head_flame_face_mask = torch.zeros(
                self.flame_faces.shape[0],
                dtype=torch.bool,
                device="cuda",
            )

        deform_mask = deformable_face_mask[pix_to_face]
        deform_mask = torch.logical_and(deform_mask, uv_mask)
        
        self.deform_mask = deform_mask.permute(0, 3, 1, 2)

        uv_mask = uv_mask.permute(0, 3, 1, 2)
        self.uv_remesh_faces = gen_uv_mesh(uv_mask)
        uv_flame_face_ids = einops.rearrange(pix_to_face.permute(0, 3, 1, 2), 'b m h w -> (b h w) m')[:, 0]
        remesh_flame_face_ids = uv_flame_face_ids[self.uv_remesh_faces]
        self.back_head_uv_remesh_face_mask = self.back_head_flame_face_mask[remesh_flame_face_ids].any(dim=1)

        # compute face area with template vertices
        # and count number of bindings
        template_verts = self.flame_verts.to(self.flame_faces.device)

        uv_remesh_verts = self.uv_remesh_flame_vertices(template_verts[None])[0]
        uv_remesh_verts = einops.rearrange(uv_remesh_verts, 'h w c -> (h w) c')

        triangles = uv_remesh_verts[self.uv_remesh_faces]

        ab = triangles[:, 1] - triangles[:, 0]
        ac = triangles[:, 2] - triangles[:, 0]
        face_area = 0.5 * torch.norm(torch.cross(ab, ac, dim=-1), dim=-1)

        gaussians_per_face = self.n_gaussians_init / face_area.sum() * face_area
        base_gaussians_per_face = gaussians_per_face.round().long().clamp(self.n_points_per_triangle)
        gaussians_per_face = base_gaussians_per_face.clone()
        if self.enable_back_head_densify:
            gaussians_per_face[self.back_head_uv_remesh_face_mask] *= self.back_head_density_multiplier

        # adjust counts per triangle according to face area
        counts = []
        binding = []
        back_head_extra_mask = []
        for i in range(gaussians_per_face.shape[0]):
            n_face_gaussians = int(gaussians_per_face[i].item())
            n_base_gaussians = int(base_gaussians_per_face[i].item())
            is_back_head_face = bool(self.back_head_uv_remesh_face_mask[i].item())
            for j in range(n_face_gaussians):
                counts.append(n_face_gaussians)
                binding.append(i)
                is_extra = is_back_head_face and j >= n_base_gaussians
                back_head_extra_mask.append(is_extra)
        self.gaussian_counts = torch.tensor(counts).float().cuda()
        self.binding = torch.tensor(binding).to(torch.int64).cuda()
        self.binding_counter = gaussians_per_face.to(torch.int32)
        self.back_head_extra_gaussian_mask = torch.tensor(back_head_extra_mask, dtype=torch.bool, device="cuda")
        self.back_head_gaussian_mask = self.back_head_uv_remesh_face_mask[self.binding]

        if self.enable_back_head_densify:
            normal_gaussians = int((~self.back_head_gaussian_mask).sum().item())
            back_head_gaussians = int(self.back_head_gaussian_mask.sum().item())
            back_head_extra_gaussians = int(self.back_head_extra_gaussian_mask.sum().item())
            print("Gaussian init normal count:", normal_gaussians)
            print("Gaussian init back-head count:", back_head_gaussians)
            print("Gaussian init back-head extra count:", back_head_extra_gaussians)
            print("Gaussian init back-head UV-remesh face count:", int(self.back_head_uv_remesh_face_mask.sum().item()))

    def create_from_pcd(self, pcd, spatial_lr_scale: float):
        if pcd is not None:
            return super().create_from_pcd(pcd, spatial_lr_scale)

        assert self.binding is not None, "Legacy 3DGS is not supported"

        self.spatial_lr_scale = spatial_lr_scale
        num_pts = self.binding.shape[0]
        fused_point_cloud = torch.tensor(np.random.random((num_pts, 3)) * 0.4).float().cuda()
        fused_color = torch.tensor(np.random.random((num_pts, 3)) / 255.0).float().cuda()

        if self.enable_back_head_densify and self.back_head_extra_gaussian_mask.any():
            hair_sh = RGB2SH(self.hair_color_rgb[None]).squeeze(0)
            fused_color[self.back_head_extra_gaussian_mask] = hair_sh

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        print("Number of points at initialisation: ", num_pts)

        scales = torch.ones((num_pts, 3), device="cuda")
        if self.gaussian_init_type == "scaled":
            scales = scales / self.gaussian_counts[:, None]
        if self.enable_back_head_densify and self.back_head_extra_gaussian_mask.any():
            scales[self.back_head_extra_gaussian_mask] *= self.back_head_extra_scale_mult
        scales = torch.log(scales)

        rots = torch.zeros((num_pts, 4), device="cuda")
        rots[:, 0] = 1

        opacity_values = 0.1 * torch.ones((num_pts, 1), dtype=torch.float, device="cuda")
        if self.enable_back_head_densify and self.back_head_extra_gaussian_mask.any():
            opacity_values[self.back_head_extra_gaussian_mask] = self.back_head_extra_opacity
        opacities = self.inverse_opacity_activation(opacity_values)

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((num_pts), device="cuda")

    def load_meshes(self, train_meshes, test_meshes, tgt_meshes):
        meshes = train_meshes + test_meshes

        if len(tgt_meshes) > 0:
            meshes = meshes + tgt_meshes
            base_rot = tgt_meshes[0]['rot']
        else:
            base_rot = meshes[0]['rot']

        T = len(meshes)

        self.flame_param = {
            'shape': torch.from_numpy(meshes[0]['shape']),
            'base_rot': torch.from_numpy(base_rot), 
            'expr': torch.zeros([T, meshes[0]['expr'].shape[0]]),
            'eye_rot': torch.zeros([T, 3]),
            'rot': torch.zeros([T, 3]),
            'tra': torch.zeros([T, 3]),
        }

        if not self.static_neck:
            self.neck_rot_offset = nn.Embedding(
                T, 3, sparse=True, _weight=torch.zeros([T, 3])
            ).cuda()

        for i, mesh in enumerate(meshes):
            self.flame_param['expr'][i] = torch.from_numpy(mesh['expr'])
            self.flame_param['eye_rot'][i] = torch.from_numpy(mesh['eye_rot'])
            self.flame_param['rot'][i] = torch.from_numpy(mesh['rot'])
            self.flame_param['tra'][i] = torch.from_numpy(mesh['tra'])
        
        for k, v in self.flame_param.items():
            self.flame_param[k] = v.float().cuda()

        self.num_timesteps = T
        self.motion_training_count = len(train_meshes)
        if self.use_motion_condition:
            if self.motion_features is None:
                print("WARNING: motion conditioning is enabled but no motion features were loaded; using zero motion conditions.")
            else:
                self.motion_features = self._align_motion_features(self.motion_features, T).cuda()
                if (
                    self.motion_condition_mode == "cross_attention_v4"
                    and not self._motion_feature_center_restored
                ):
                    self._set_motion_feature_center(
                        self.motion_features,
                        source="all_indexed_features",
                    )
                self.motion_neutral_index = self._select_neutral_motion_index()
                if self.motion_nodeform_condition == "neutral":
                    print(
                        "Selected neutral nodeform condition frame:",
                        int(self.motion_neutral_index),
                        "score:",
                        float(self._neutral_motion_scores()[self.motion_neutral_index].detach().cpu()),
                    )

    def _load_motion_feature_object(self, motion_feature_path):
        path = Path(motion_feature_path)
        if not path.exists():
            raise FileNotFoundError(f"motion feature file does not exist: {path}")

        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        except Exception:
            data = np.load(path, allow_pickle=True)

        if isinstance(data, dict):
            for key in ("motion_features", "features", "x"):
                if key in data:
                    data = data[key]
                    break
        return data

    def _as_motion_feature_matrix(self, motion_features):
        if isinstance(motion_features, np.lib.npyio.NpzFile):
            if len(motion_features.files) != 1:
                raise ValueError(
                    f"Cannot infer motion feature array from npz keys: {motion_features.files}"
                )
            motion_features = motion_features[motion_features.files[0]]

        motion_features = torch.as_tensor(motion_features).float()
        if motion_features.ndim == 4:
            if motion_features.shape[0] != 1:
                raise ValueError(
                    f"Expected motion features with batch size 1, got {tuple(motion_features.shape)}"
                )
            motion_features = motion_features[0]
        if motion_features.ndim == 3:
            motion_features = motion_features.reshape(motion_features.shape[0], -1)
        elif motion_features.ndim == 2:
            pass
        elif motion_features.ndim == 1:
            motion_features = motion_features[None]
        else:
            raise ValueError(
                f"Unsupported motion feature shape: {tuple(motion_features.shape)}"
            )

        if motion_features.shape[-1] != 512:
            raise ValueError(
                f"Expected per-frame motion feature dim 512, got {motion_features.shape[-1]}"
            )
        return motion_features

    def load_motion_features(self, motion_feature_path):
        motion_features = self._load_motion_feature_object(motion_feature_path)
        self.motion_features = self._as_motion_feature_matrix(motion_features)
        if (
            self.motion_condition_mode == "cross_attention_v4"
            and not self._motion_feature_center_restored
        ):
            self._set_motion_feature_center(
                self.motion_features,
                source="loaded_training_features",
            )
        print(
            "Loaded Xnemo motion features:",
            motion_feature_path,
            "shape:",
            tuple(self.motion_features.shape),
        )

    def _set_motion_feature_center(self, motion_features, source):
        features = self._as_motion_feature_matrix(motion_features).detach().float()
        if features.shape[0] == 0:
            raise ValueError("Cannot compute a motion feature center from an empty tensor.")
        center = features.mean(dim=0, keepdim=True)
        centered = features - center
        sample_energy = features.square().sum(dim=1).mean().clamp_min(1e-12)
        common_energy = center.square().sum()
        self.motion_feature_center = center.cpu()
        self.motion_feature_center_source = str(source)
        self.motion_feature_common_energy_ratio = float(
            (common_energy / sample_energy).cpu()
        )
        self.motion_feature_centered_rms = float(
            centered.square().mean().sqrt().cpu()
        )
        print(
            "Xnemo condition centering:",
            f"mode={self.motion_cross_attention_centering}",
            f"source={self.motion_feature_center_source}",
            f"common_energy_ratio={self.motion_feature_common_energy_ratio:.6f}",
            f"centered_rms={self.motion_feature_centered_rms:.6f}",
        )

    def _condition_motion_feature(self, motion_feature):
        if (
            self.motion_condition_mode != "cross_attention_v4"
            or self.motion_cross_attention_centering == "none"
        ):
            return motion_feature
        center = self.motion_feature_center.to(
            device=motion_feature.device,
            dtype=motion_feature.dtype,
        )
        nonzero = motion_feature.abs().sum(dim=-1, keepdim=True) > 0
        return (motion_feature - center) * nonzero.to(dtype=motion_feature.dtype)

    def _align_motion_features(self, motion_features, n_timesteps):
        n_motion = motion_features.shape[0]
        if n_motion == n_timesteps:
            return motion_features

        if self.motion_feature_align == "strict":
            raise ValueError(
                "Motion feature frame count must match FLAME timesteps when "
                f"motion_feature_align='strict': motion={n_motion}, flame={n_timesteps}. "
                "Use aligned features, or set motion_feature_align to 'interpolate' or 'truncate'."
            )
        if self.motion_feature_align == "interpolate":
            motion_features = F.interpolate(
                motion_features.T[None],
                size=n_timesteps,
                mode="linear",
                align_corners=False,
            )[0].T
            print(f"Interpolated motion features from {n_motion} to {n_timesteps} frames.")
            return motion_features
        if self.motion_feature_align == "truncate":
            if n_motion < n_timesteps:
                pad = motion_features[[-1]].repeat(n_timesteps - n_motion, 1)
                motion_features = torch.cat([motion_features, pad], dim=0)
            else:
                motion_features = motion_features[:n_timesteps]
            print(f"Truncated/padded motion features from {n_motion} to {n_timesteps} frames.")
            return motion_features

        raise ValueError(
            f"Unsupported motion_feature_align={self.motion_feature_align!r}. "
            "Use 'strict', 'interpolate', or 'truncate'."
        )

    def _prepare_motion_feature(self, batch_size, device, dtype, motion_feature_override=None):
        if (
            self._is_training_mode
            and self.motion_condition_mode in (
                "cross_attention_v2",
                "cross_attention_v3",
                "cross_attention_v4",
            )
            and self._cross_attention_base_pretrain_active
        ):
            motion_feature = torch.zeros((batch_size, 512), device=device, dtype=dtype)
        elif motion_feature_override is None:
            motion_feature = self._current_motion_feature(batch_size, device, dtype)
        else:
            motion_feature = self._coerce_motion_feature(
                motion_feature_override,
                batch_size,
                device,
                dtype,
                label="override motion feature",
            )
        self._last_prepared_motion_feature = motion_feature.detach()
        condition_feature = self._condition_motion_feature(motion_feature)
        self._last_condition_motion_feature = condition_feature.detach()
        return condition_feature

    @staticmethod
    def _coerce_motion_feature(motion_feature, batch_size, device, dtype, label):
        motion_feature = motion_feature.to(device=device, dtype=dtype)
        if motion_feature.ndim == 1:
            motion_feature = motion_feature[None]
        if motion_feature.ndim != 2 or motion_feature.shape[-1] != 512:
            raise ValueError(
                f"Expected {label} shape [B, 512], got {tuple(motion_feature.shape)}"
            )
        if motion_feature.shape[0] == 1 and batch_size != 1:
            motion_feature = motion_feature.expand(batch_size, -1)
        if motion_feature.shape[0] != batch_size:
            raise ValueError(
                f"{label.capitalize()} batch {motion_feature.shape[0]} must match {batch_size}"
            )
        return motion_feature

    def _neutral_motion_scores(self):
        expr = self.flame_param["expr"].detach()
        eye_rot = self.flame_param["eye_rot"].detach()
        return expr.norm(dim=1) + eye_rot.norm(dim=1)

    def _select_neutral_motion_index(self):
        if self.flame_param is None or self.motion_features is None:
            return 0
        scores = self._neutral_motion_scores()
        if scores.numel() == 0:
            return 0
        return int(torch.argmin(scores).item())

    def _current_motion_feature(self, batch_size, device, dtype):
        if not self.use_motion_condition:
            return None

        if self._motion_feature_override is not None:
            return self._coerce_motion_feature(
                self._motion_feature_override,
                batch_size,
                device,
                dtype,
                label="model motion feature override",
            )

        if self.motion_features is None:
            motion_feature = torch.zeros(
                (batch_size, 512),
                dtype=dtype,
                device=device,
            )
        else:
            if self.timestep is None:
                timestep = 0
            else:
                timestep = int(self.timestep)
            if timestep >= self.motion_features.shape[0]:
                raise IndexError(
                    f"motion feature timestep {timestep} is out of range for "
                    f"{self.motion_features.shape[0]} frames"
                )
            if self._is_training_mode and self.motion_condition_runtime_shuffle == "batch":
                indices = torch.randint(
                    0,
                    self.motion_features.shape[0],
                    (batch_size,),
                    device=self.motion_features.device,
                )
                motion_feature = self.motion_features[indices].to(device=device, dtype=dtype)
            else:
                motion_feature = self.motion_features[[timestep]].to(
                    device=device,
                    dtype=dtype,
                )
                if batch_size != 1:
                    motion_feature = motion_feature.expand(batch_size, -1)

        if motion_feature.ndim != 2 or motion_feature.shape[-1] != 512:
            raise ValueError(
                f"Expected current motion feature shape [B, 512], got {tuple(motion_feature.shape)}"
            )
        return motion_feature

    def _sample_mismatched_motion_feature(self, current_raw, device, dtype):
        if self.motion_features is None:
            self._last_motion_mismatch_index = None
            self._last_motion_mismatch_flame_distance = 0.0
            self._last_motion_mismatch_condition_cosine = 1.0
            return current_raw.detach().clone()

        n_motion = int(self.motion_features.shape[0])
        if self._is_training_mode and self.motion_training_count is not None:
            n_motion = min(n_motion, int(self.motion_training_count))
        if n_motion < 2:
            self._last_motion_mismatch_index = None
            self._last_motion_mismatch_flame_distance = 0.0
            self._last_motion_mismatch_condition_cosine = 1.0
            return current_raw.detach().clone()
        timestep = int(self.timestep) if self.timestep is not None else 0
        timestep = min(max(timestep, 0), n_motion - 1)
        candidate_count = min(
            self.motion_cross_attention_mismatch_candidates,
            n_motion - 1,
        )

        flame_distances = None
        if (
            self.motion_cross_attention_mismatch_selection == "flame_nearest"
            and self.flame_param is not None
        ):
            signature = torch.cat(
                [self.flame_param["expr"], self.flame_param["eye_rot"]],
                dim=1,
            ).detach()
            signature = signature[:n_motion].to(device=device, dtype=torch.float32)
            signature_scale = signature.std(dim=0, unbiased=False).clamp_min(1e-4)
            flame_distances = (
                ((signature - signature[[timestep]]) / signature_scale)
                .square()
                .mean(dim=1)
            )
            flame_distances[timestep] = torch.inf
            candidate_indices = torch.topk(
                flame_distances,
                k=candidate_count,
                largest=False,
            ).indices
        else:
            candidate_indices = torch.randperm(n_motion, device=device)
            candidate_indices = candidate_indices[candidate_indices != timestep][
                :candidate_count
            ]

        candidate_raw = self.motion_features.index_select(
            0,
            candidate_indices.to(device=self.motion_features.device),
        ).to(
            device=device,
            dtype=dtype,
        )
        candidate_condition = self._condition_motion_feature(candidate_raw)
        current_condition = self._condition_motion_feature(current_raw[:1])
        candidate_cosine = F.cosine_similarity(
            candidate_condition,
            current_condition.expand_as(candidate_condition),
            dim=1,
            eps=1e-8,
        )
        selected_local = int(torch.argmin(candidate_cosine).item())
        selected_index = int(candidate_indices[selected_local].item())
        selected_raw = candidate_raw[[selected_local]]
        if current_raw.shape[0] != 1:
            selected_raw = selected_raw.expand(current_raw.shape[0], -1)

        self._last_motion_mismatch_index = selected_index
        self._last_motion_mismatch_condition_cosine = float(
            candidate_cosine[selected_local].detach().cpu()
        )
        self._last_motion_mismatch_flame_distance = (
            float(flame_distances[selected_index].detach().cpu())
            if flame_distances is not None else -1.0
        )
        return selected_raw

    def _nodeform_motion_feature(self, batch_size, device, dtype, deform_motion_feature):
        if not self.use_motion_condition:
            return None
        if self.motion_nodeform_condition == "zero" or self.motion_features is None:
            return torch.zeros((batch_size, 512), dtype=dtype, device=device)
        if self.motion_nodeform_condition == "mean":
            feature = self.motion_features.mean(dim=0, keepdim=True).to(device=device, dtype=dtype)
        elif self.motion_nodeform_condition == "neutral":
            neutral_idx = self.motion_neutral_index
            if neutral_idx is None:
                neutral_idx = self._select_neutral_motion_index()
                self.motion_neutral_index = neutral_idx
            feature = self.motion_features[[neutral_idx]].to(device=device, dtype=dtype)
        else:
            raise RuntimeError(f"Unexpected motion_nodeform_condition={self.motion_nodeform_condition!r}")
        if batch_size != 1:
            feature = feature.expand(batch_size, -1)
        return feature

    def _current_motion_condition_map(self, uv_offsets):
        if not self.use_legacy_motion_concat:
            return None

        motion_feature = self._current_motion_feature(
            uv_offsets.shape[0],
            uv_offsets.device,
            uv_offsets.dtype,
        )
        motion_condition = self.motion_condition_proj(motion_feature)
        return motion_condition[:, :, None, None].expand(
            -1, -1, uv_offsets.shape[-2], uv_offsets.shape[-1]
        )

    def get_bbox_center(self):
        bbox_center = (self.verts.max(dim=1)[0] + self.verts.min(dim=1)[0]) / 2.
        return bbox_center

    def eval(self):
        self._is_training_mode = False
        self.deform_net.eval()
        if self.motion_condition_proj is not None:
            self.motion_condition_proj.eval()

    def train(self):
        self._is_training_mode = True
        self.deform_net.train()
        if self.motion_condition_proj is not None:
            self.motion_condition_proj.train()

    def set_motion_feature_override(self, motion_feature):
        self._motion_feature_override = motion_feature

    def clear_motion_feature_override(self):
        self._motion_feature_override = None

    def select_mesh_by_timestep(self, timestep):
        self.timestep = timestep
        
        base_rot = self.flame_param["base_rot"][None]
        curr_rot = self.flame_param["rot"][[timestep]]
        relative_rot = roma.rotvec_to_rotmat(curr_rot).inverse() @ roma.rotvec_to_rotmat(base_rot)
        relative_rot = roma.rotmat_to_rotvec(relative_rot)

        # limit neck rotation to not break the gaussians (hacky)
        MAX_NECK_ROT = 0.15
        relative_rot = torch.tanh(relative_rot / MAX_NECK_ROT) * MAX_NECK_ROT

        if not self.static_neck:
            # allow the neck to rotate during training to correct generated images
            neck_rot_offset = self.neck_rot_offset(
                torch.tensor([timestep], dtype=torch.long, device=relative_rot.device)
            )
            relative_rot = relative_rot + neck_rot_offset

        # compute flame for deformed and neutral mesh (with neck rotations)
        verts, _ = self.flame_model({
            "shape": self.flame_param["shape"],
            "expr": self.flame_param["expr"][[timestep]],
            "rot": self.flame_param["rot"][[timestep]],
            "tra": self.flame_param["tra"][[timestep]],
            "eye_rot": self.flame_param["eye_rot"][[timestep]],
            "neck_rot": relative_rot,
        })
        # convert from p3d to opencv convention
        verts[..., 1] = -verts[..., 1]
        verts[..., 2] = -verts[..., 2]
        self.current_flame_verts_for_region_masks = verts.detach()

        neutral_verts, _ = self.flame_model({
            "shape": self.flame_param["shape"],
            "expr": self.flame_param["expr"][[timestep]] * 0.,
            "rot": self.flame_param["rot"][[timestep]],
            "tra": self.flame_param["tra"][[timestep]],
            "eye_rot": self.flame_param["eye_rot"][[timestep]] * 0.,
            "neck_rot": relative_rot,
        })
        # convert from p3d to opencv convention
        neutral_verts[..., 1] = -neutral_verts[..., 1]
        neutral_verts[..., 2] = -neutral_verts[..., 2]

        offsets = verts - neutral_verts

        self.update_mesh_properties(verts, offsets)

    def uv_remesh_flame_vertices(self, verts):
        verts_packed = verts[:, self.flame_faces]
        # remesh vertices
        uv_px_verts = self.vert_shader._rasterize_property(verts_packed, self.fragments)
        uv_px_verts = uv_px_verts.squeeze(3)

        return uv_px_verts

    def _regularize_cross_attention_uv_input(self, uv_offsets):
        self._last_cross_attention_uv_dropout_fraction = 0.0
        self._last_cross_attention_uv_noise_std = 0.0
        if (
            not self._is_training_mode
            or self.motion_condition_mode not in (
                "cross_attention",
                "cross_attention_v2",
                "cross_attention_v3",
                "cross_attention_v4",
            )
            or (
                self.motion_condition_mode in (
                    "cross_attention_v2",
                    "cross_attention_v3",
                    "cross_attention_v4",
                )
                and self._cross_attention_base_pretrain_active
            )
        ):
            return uv_offsets

        out = uv_offsets
        dropout_prob = self.motion_cross_attention_uv_dropout_prob
        if dropout_prob > 0.0:
            keep_mask = (
                torch.rand((out.shape[0], 1, 1, 1), device=out.device, dtype=out.dtype)
                >= dropout_prob
            ).to(dtype=out.dtype)
            dropped_mask = 1.0 - keep_mask
            out = out * (keep_mask + dropped_mask * self.motion_cross_attention_uv_dropout_scale)
            self._last_cross_attention_uv_dropout_fraction = float(dropped_mask.mean().detach().cpu())

        noise_std = self.motion_cross_attention_uv_noise_std
        if noise_std > 0.0:
            out = out + torch.randn_like(out) * noise_std
            self._last_cross_attention_uv_noise_std = float(noise_std)
        return out

    def forward_unet(self, uv_offsets, motion_feature_override=None, save_debug=True):
        if self.use_expr_mask:
            # import pdb; pdb.set_trace()
            # use mask to prevent deformations in undesired regions
            uv_offsets = uv_offsets * self.uv_mask

        uv_offsets_for_deform = self._regularize_cross_attention_uv_input(uv_offsets.detach())
        deform_parts = [uv_offsets_for_deform, self.pos_enc[None]]
        nodeform_parts = [torch.zeros_like(uv_offsets), self.pos_enc[None]]
        motion_condition = self._current_motion_condition_map(uv_offsets)
        if motion_condition is not None:
            deform_parts.append(motion_condition)
            nodeform_feature = self._nodeform_motion_feature(
                uv_offsets.shape[0],
                uv_offsets.device,
                uv_offsets.dtype,
                None,
            )
            nodeform_condition = self.motion_condition_proj(nodeform_feature)
            nodeform_parts.append(
                nodeform_condition[:, :, None, None].expand(
                    -1, -1, uv_offsets.shape[-2], uv_offsets.shape[-1]
                )
            )

        deform_input = torch.cat(deform_parts, dim=1)
        nodeform_input = torch.cat(nodeform_parts, dim=1)

        unet_input = torch.cat([deform_input, nodeform_input], dim=0)

        deform_net = self.deform_net.module if hasattr(self.deform_net, "module") else self.deform_net
        self.spatial_residual_v2_base_actual_full = None
        self.spatial_residual_v2_delta_actual_full = None
        self.spatial_residual_v2_deform_output = None
        self.spatial_residual_v2_nodeform_output = None
        self.cross_attention_v3_base_actual_full = None
        self.cross_attention_v3_delta_actual_full = None
        self.cross_attention_v3_deform_output = None
        self.cross_attention_v3_nodeform_output = None
        self.cross_attention_v4_base_actual_full = None
        self.cross_attention_v4_delta_actual_full = None
        self.cross_attention_v4_mismatch_delta_actual_full = None
        self.cross_attention_v4_deform_output = None
        self.cross_attention_v4_nodeform_output = None
        self.cross_attention_v4_mismatch_deform_output = None
        self.deform_base_output = None
        self.neutral_base_output = None
        cross_attention_base_output_norm = None
        cross_attention_v4_mismatch_output_norm = None

        if self.use_conditional_norm:
            motion_feature = self._prepare_motion_feature(
                uv_offsets.shape[0],
                uv_offsets.device,
                uv_offsets.dtype,
                motion_feature_override=motion_feature_override,
            )
            nodeform_feature = self._nodeform_motion_feature(
                uv_offsets.shape[0],
                uv_offsets.device,
                uv_offsets.dtype,
                motion_feature,
            )
            if self.motion_condition_mode == "cross_attention_v4":
                nodeform_feature = self._condition_motion_feature(nodeform_feature)
            condition_input = torch.cat(
                [motion_feature, nodeform_feature],
                dim=0,
            )
            if condition_input.shape[0] != unet_input.shape[0]:
                raise ValueError(
                    f"Condition batch {condition_input.shape[0]} must match U-Net batch {unet_input.shape[0]}"
                )
            if (
                self.motion_condition_mode in ("cross_attention_v3", "cross_attention_v4")
                and self._is_training_mode
            ):
                # Compare against the frozen zero-condition path using the exact same UV input.
                # Running this first leaves per-site attention statistics from the conditioned pass.
                with torch.no_grad():
                    cross_attention_base_output_norm = self.deform_net(
                        unet_input,
                        condition=torch.zeros_like(condition_input),
                    )
                if (
                    self.motion_condition_mode == "cross_attention_v4"
                    and self.motion_cross_attention_mismatch_enabled
                    and not self._cross_attention_base_pretrain_active
                ):
                    current_raw = self._last_prepared_motion_feature.to(
                        device=uv_offsets.device,
                        dtype=uv_offsets.dtype,
                    )
                    mismatch_raw = self._sample_mismatched_motion_feature(
                        current_raw,
                        uv_offsets.device,
                        uv_offsets.dtype,
                    )
                    mismatch_feature = self._condition_motion_feature(mismatch_raw)
                    mismatch_condition_input = torch.cat(
                        [mismatch_feature, nodeform_feature],
                        dim=0,
                    )
                    cross_attention_v4_mismatch_output_norm = self.deform_net(
                        unet_input,
                        condition=mismatch_condition_input,
                    )
            unet_output_norm = self.deform_net(unet_input, condition=condition_input)
            base_output_norm = (
                deform_net.get_base_output()
                if hasattr(deform_net, "get_base_output") else None
            )
            condition_residual_norm = (
                deform_net.get_condition_residual()
                if hasattr(deform_net, "get_condition_residual") else None
            )
            if (
                self.motion_condition_mode == "spatial_residual_branch_v2"
                and base_output_norm is not None
                and condition_residual_norm is not None
            ):
                self.spatial_residual_v2_base_actual_full = base_output_norm * STD_DEFORM
                self.spatial_residual_v2_delta_actual_full = condition_residual_norm * STD_DEFORM
            if cross_attention_base_output_norm is not None:
                base_actual_full = cross_attention_base_output_norm * STD_DEFORM
                delta_actual_full = (
                    unet_output_norm - cross_attention_base_output_norm
                ) * STD_DEFORM
                if self.motion_condition_mode == "cross_attention_v3":
                    self.cross_attention_v3_base_actual_full = base_actual_full
                    self.cross_attention_v3_delta_actual_full = delta_actual_full
                else:
                    self.cross_attention_v4_base_actual_full = base_actual_full
                    self.cross_attention_v4_delta_actual_full = delta_actual_full
                    if cross_attention_v4_mismatch_output_norm is not None:
                        self.cross_attention_v4_mismatch_delta_actual_full = (
                            cross_attention_v4_mismatch_output_norm
                            - cross_attention_base_output_norm
                        ) * STD_DEFORM
            unet_output = unet_output_norm * STD_DEFORM
        else:
            unet_output = self.deform_net(unet_input) * STD_DEFORM # unnormalization!

        deform_output, nodeform_output = unet_output.chunk(2, dim=0)
        if self.spatial_residual_v2_base_actual_full is not None:
            base_deform_output, base_nodeform_output = self.spatial_residual_v2_base_actual_full.chunk(2, dim=0)
            delta_deform_output, delta_nodeform_output = self.spatial_residual_v2_delta_actual_full.chunk(2, dim=0)
            self.neutral_base_output = base_nodeform_output
            self.spatial_residual_v2_nodeform_output = delta_nodeform_output
        if self.cross_attention_v3_base_actual_full is not None:
            v3_base_deform_output, v3_base_nodeform_output = self.cross_attention_v3_base_actual_full.chunk(2, dim=0)
            _, v3_delta_nodeform_output = self.cross_attention_v3_delta_actual_full.chunk(2, dim=0)
            self.neutral_base_output = v3_base_nodeform_output
            self.cross_attention_v3_nodeform_output = v3_delta_nodeform_output
        if self.cross_attention_v4_base_actual_full is not None:
            v4_base_deform_output, v4_base_nodeform_output = (
                self.cross_attention_v4_base_actual_full.chunk(2, dim=0)
            )
            _, v4_delta_nodeform_output = (
                self.cross_attention_v4_delta_actual_full.chunk(2, dim=0)
            )
            self.neutral_base_output = v4_base_nodeform_output
            self.cross_attention_v4_nodeform_output = v4_delta_nodeform_output
            if self.cross_attention_v4_mismatch_delta_actual_full is not None:
                (
                    v4_mismatch_delta_deform_output,
                    v4_mismatch_delta_nodeform_output,
                ) = self.cross_attention_v4_mismatch_delta_actual_full.chunk(2, dim=0)

        # set deform mask places to neutral output so that it cannot deform
        deform_output = self.deform_mask * deform_output + torch.logical_not(self.deform_mask) * nodeform_output
        deform_output = self._apply_back_static_mask(deform_output, nodeform_output)
        if self.spatial_residual_v2_base_actual_full is not None:
            base_deform_masked = self.deform_mask * base_deform_output + torch.logical_not(self.deform_mask) * base_nodeform_output
            base_deform_masked = self._apply_back_static_mask(base_deform_masked, base_nodeform_output)
            self.deform_base_output = base_deform_masked
            self.spatial_residual_v2_deform_output = deform_output - base_deform_masked
        if self.cross_attention_v3_base_actual_full is not None:
            v3_base_deform_masked = (
                self.deform_mask * v3_base_deform_output
                + torch.logical_not(self.deform_mask) * v3_base_nodeform_output
            )
            v3_base_deform_masked = self._apply_back_static_mask(
                v3_base_deform_masked,
                v3_base_nodeform_output,
            )
            self.deform_base_output = v3_base_deform_masked
            self.cross_attention_v3_deform_output = deform_output - v3_base_deform_masked
        if self.cross_attention_v4_base_actual_full is not None:
            v4_base_deform_pre_back = (
                self.deform_mask * v4_base_deform_output
                + torch.logical_not(self.deform_mask) * v4_base_nodeform_output
            )
            v4_base_deform_masked = self._apply_back_static_mask(
                v4_base_deform_pre_back,
                v4_base_nodeform_output,
            )
            self.deform_base_output = v4_base_deform_masked
            self.cross_attention_v4_deform_output = deform_output - v4_base_deform_masked
            if self.cross_attention_v4_mismatch_delta_actual_full is not None:
                v4_mismatch_delta_masked = (
                    self.deform_mask * v4_mismatch_delta_deform_output
                    + torch.logical_not(self.deform_mask)
                    * v4_mismatch_delta_nodeform_output
                )
                v4_mismatch_output_masked = self._apply_back_static_mask(
                    v4_base_deform_pre_back + v4_mismatch_delta_masked,
                    v4_base_nodeform_output + v4_mismatch_delta_nodeform_output,
                )
                self.cross_attention_v4_mismatch_deform_output = (
                    v4_mismatch_output_masked - v4_base_deform_masked
                )
        if save_debug:
            self._maybe_save_deform_debug(deform_output, nodeform_output)

        return deform_output, nodeform_output

    def _apply_back_static_mask(self, deform_output, nodeform_output):
        if not self.enable_back_static_mask or self.back_static_mask is None:
            return deform_output

        mask = self.back_static_mask.to(device=deform_output.device, dtype=deform_output.dtype)
        if mask.shape[-2:] != deform_output.shape[-2:]:
            mask = F.interpolate(mask, size=deform_output.shape[-2:], mode="bilinear", align_corners=False)

        if self.back_static_mode == "zero":
            static_output = torch.zeros_like(deform_output)
        else:
            static_output = nodeform_output

        return deform_output * (1. - mask) + static_output * mask

    def _maybe_save_deform_debug(self, deform_output, nodeform_output):
        if not self.enable_back_static_mask or not self.save_deform_debug:
            return

        self._deform_debug_counter += 1
        if self.deform_debug_interval > 0 and self._deform_debug_counter % self.deform_debug_interval != 0:
            return

        self.deform_debug_dir.mkdir(parents=True, exist_ok=True)
        timestep = getattr(self, "timestep", "unknown")
        prefix = self.deform_debug_dir / f"deform_{self._deform_debug_counter:06d}_t{timestep}"
        self._save_deform_heatmap(deform_output, prefix.with_name(prefix.name + "_staticized.png"))
        self._save_deform_heatmap(nodeform_output, prefix.with_name(prefix.name + "_neutral.png"))
        if self.back_static_mask is not None:
            mask = self.back_static_mask.detach().float().cpu()[0, 0].numpy()
            Image.fromarray((mask * 255.).clip(0, 255).astype(np.uint8)).save(
                prefix.with_name(prefix.name + "_mask.png")
            )

    def _save_deform_heatmap(self, deform_output, path: Path):
        magnitude = deform_output.detach().float().norm(dim=1)[0]
        magnitude = magnitude - magnitude.min()
        denom = magnitude.max().clamp_min(1e-8)
        magnitude = (magnitude / denom).cpu().numpy()
        heatmap = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
        heatmap[..., 0] = (magnitude * 255.).clip(0, 255).astype(np.uint8)
        heatmap[..., 1] = ((1. - np.abs(magnitude - 0.5) * 2.) * 255.).clip(0, 255).astype(np.uint8)
        heatmap[..., 2] = ((1. - magnitude) * 255.).clip(0, 255).astype(np.uint8)
        Image.fromarray(heatmap).save(path)

    @staticmethod
    def _tensor_diff_stats(a, b):
        diff = (a - b).detach().abs()
        return {
            "max_abs": float(diff.max().cpu()),
            "mean_abs": float(diff.mean().cpu()),
        }

    @staticmethod
    def _param_grad_norm(parameters):
        total = 0.0
        for param in parameters:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            total += float((grad * grad).sum().cpu())
        return total ** 0.5

    def collect_condition_monitor_stats(self, include_sensitivity=True):
        if not self.use_conditional_norm:
            return {}

        stats = {}
        deform_net = self.deform_net.module if hasattr(self.deform_net, "module") else self.deform_net
        if hasattr(deform_net, "get_adain_site_table"):
            site_table = deform_net.get_adain_site_table()
            stats["condition/site_count"] = float(len(site_table))
        if hasattr(deform_net, "get_cross_attention_site_table"):
            cross_attention_site_table = deform_net.get_cross_attention_site_table()
            stats["condition/cross_attention/site_count"] = float(len(cross_attention_site_table))
            stats["condition/cross_attention/param_count"] = float(
                sum(row.get("parameters", 0) for row in cross_attention_site_table)
            )
            stats["condition/cross_attention/input_uv_dropout_fraction"] = float(
                self._last_cross_attention_uv_dropout_fraction
            )
            stats["condition/cross_attention/input_uv_noise_std"] = float(
                self._last_cross_attention_uv_noise_std
            )
            stats["condition/cross_attention/base_pretrain_active"] = float(
                self._cross_attention_base_pretrain_active
            )
            stats["condition/cross_attention/base_lr_scale"] = float(
                self._cross_attention_base_lr_scale
            )
            stats["condition/cross_attention/condition_lr_scale"] = float(
                self._cross_attention_condition_lr_scale
            )
            stats["condition/cross_attention/adapter_only"] = float(
                self.motion_cross_attention_adapter_only
            )
            if self.motion_condition_mode == "cross_attention_v4":
                stats["condition/cross_attention_v4/feature_common_energy_ratio"] = float(
                    self.motion_feature_common_energy_ratio
                )
                stats["condition/cross_attention_v4/feature_centered_rms"] = float(
                    self.motion_feature_centered_rms
                )
                stats["condition/cross_attention_v4/mismatch_enabled"] = float(
                    self.motion_cross_attention_mismatch_enabled
                )
                stats["condition/cross_attention_v4/mismatch_index"] = float(
                    self._last_motion_mismatch_index
                    if self._last_motion_mismatch_index is not None else -1
                )
                stats["condition/cross_attention_v4/mismatch_flame_distance"] = float(
                    self._last_motion_mismatch_flame_distance
                )
                stats["condition/cross_attention_v4/mismatch_condition_cosine"] = float(
                    self._last_motion_mismatch_condition_cosine
                )
                if hasattr(self, "_last_prepared_motion_feature"):
                    raw_feature = self._last_prepared_motion_feature.detach().float()
                    stats["condition/cross_attention_v4/current_raw_rms"] = float(
                        raw_feature.square().mean().sqrt().cpu()
                    )
                if hasattr(self, "_last_condition_motion_feature"):
                    condition_feature = self._last_condition_motion_feature.detach().float()
                    stats["condition/cross_attention_v4/current_centered_rms"] = float(
                        condition_feature.square().mean().sqrt().cpu()
                    )
        if hasattr(deform_net, "get_condition_stats"):
            for module_name, site_stats in deform_net.get_condition_stats().items():
                if not site_stats:
                    continue
                site = str(site_stats.get("site", module_name)).replace(".", "/")
                for key in ("gamma_mean", "gamma_std", "beta_mean", "beta_std", "feature_delta_mean_abs", "feature_delta_max_abs"):
                    if key in site_stats:
                        stats[f"condition/{site}/{key}"] = float(site_stats[key])
        if hasattr(deform_net, "get_spatial_residual_v2_stats"):
            for key, value in deform_net.get_spatial_residual_v2_stats().items():
                stats[f"condition/spatial_residual_v2/{key}"] = float(value)
        if hasattr(deform_net, "get_spatial_residual_v2_grad_stats"):
            stats.update(deform_net.get_spatial_residual_v2_grad_stats())
        if hasattr(deform_net, "get_cross_attention_stats"):
            for module_name, site_stats in deform_net.get_cross_attention_stats().items():
                if not site_stats:
                    continue
                site = str(site_stats.get("site", module_name)).replace(".", "/")
                for key in (
                    "gate",
                    "num_tokens",
                    "hidden_dim",
                    "attention_dim",
                    "num_heads",
                    "token_mean",
                    "token_std",
                    "attention_entropy_mean",
                    "attention_entropy_std",
                    "attention_max_mean",
                    "logit_scale",
                    "normalize_qk",
                    "direct_token_projection",
                    "use_spatial_position",
                    "key_token_bias_std",
                    "feature_delta_mean_abs",
                    "feature_delta_max_abs",
                ):
                    if key in site_stats:
                        stats[f"condition/cross_attention/{site}/{key}"] = float(site_stats[key])
        if hasattr(deform_net, "get_cross_attention_grad_stats"):
            stats.update(deform_net.get_cross_attention_grad_stats())

        for module_name, module in deform_net.named_modules():
            if hasattr(module, "head") and hasattr(module, "site_name"):
                site = str(module.site_name).replace(".", "/")
                stats[f"condition/{site}/head_grad_norm"] = self._param_grad_norm(module.head.parameters())
            if module.__class__.__name__ == "SharedConditionTrunk":
                stats["condition/shared_trunk_grad_norm"] = self._param_grad_norm(module.parameters())

        if hasattr(self, "deform_output") and self.deform_output is not None:
            stats["condition/deform_output_mean_abs"] = float(self.deform_output.detach().abs().mean().cpu())
            stats["condition/deform_output_l2"] = float(self.deform_output.detach().norm().cpu())
        if hasattr(self, "neutral_output") and self.neutral_output is not None:
            stats["condition/nodeform_output_mean_abs"] = float(self.neutral_output.detach().abs().mean().cpu())
            stats["condition/nodeform_output_l2"] = float(self.neutral_output.detach().norm().cpu())
        if self.spatial_residual_v2_deform_output is not None:
            delta = self.spatial_residual_v2_deform_output.detach()
            base = (
                self.deform_base_output.detach()
                if self.deform_base_output is not None else torch.zeros_like(delta)
            )
            stats["condition/spatial_residual_v2/delta_actual_mean_abs"] = float(delta.abs().mean().cpu())
            stats["condition/spatial_residual_v2/delta_actual_std"] = float(delta.std(unbiased=False).cpu())
            stats["condition/spatial_residual_v2/delta_actual_max_abs"] = float(delta.abs().max().cpu())
            stats["condition/spatial_residual_v2/base_actual_mean_abs"] = float(base.abs().mean().cpu())
            stats["condition/spatial_residual_v2/delta_over_base"] = float(
                (delta.abs().mean() / (base.abs().mean() + 1e-8)).cpu()
            )
            flat = delta.flatten(2)
            stats["condition/spatial_residual_v2/delta_actual_spatial_std"] = float(
                flat.std(dim=-1, unbiased=False).mean().cpu()
            )
        if self.cross_attention_v3_deform_output is not None:
            delta = self.cross_attention_v3_deform_output.detach()
            base = (
                self.deform_base_output.detach()
                if self.deform_base_output is not None else torch.zeros_like(delta)
            )
            delta_rms = delta.square().mean().sqrt()
            base_rms = base.square().mean().sqrt()
            stats["condition/cross_attention_v3/delta_actual_mean_abs"] = float(
                delta.abs().mean().cpu()
            )
            stats["condition/cross_attention_v3/delta_actual_rms"] = float(delta_rms.cpu())
            stats["condition/cross_attention_v3/delta_actual_max_abs"] = float(
                delta.abs().max().cpu()
            )
            stats["condition/cross_attention_v3/base_actual_mean_abs"] = float(
                base.abs().mean().cpu()
            )
            stats["condition/cross_attention_v3/base_actual_rms"] = float(base_rms.cpu())
            stats["condition/cross_attention_v3/delta_over_base_rms"] = float(
                (delta_rms / (base_rms + 1e-8)).cpu()
            )
            stats["condition/cross_attention_v3/nodeform_delta_actual_max_abs"] = float(
                self.cross_attention_v3_nodeform_output.detach().abs().max().cpu()
            )
        if self.cross_attention_v4_deform_output is not None:
            delta = self.cross_attention_v4_deform_output.detach()
            base = (
                self.deform_base_output.detach()
                if self.deform_base_output is not None else torch.zeros_like(delta)
            )
            mismatch_delta = (
                self.cross_attention_v4_mismatch_deform_output.detach()
                if self.cross_attention_v4_mismatch_deform_output is not None
                else torch.zeros_like(delta)
            )
            delta_rms = delta.square().mean().sqrt()
            base_rms = base.square().mean().sqrt()
            mismatch_rms = mismatch_delta.square().mean().sqrt()
            stats["condition/cross_attention_v4/delta_actual_mean_abs"] = float(
                delta.abs().mean().cpu()
            )
            stats["condition/cross_attention_v4/delta_actual_rms"] = float(
                delta_rms.cpu()
            )
            stats["condition/cross_attention_v4/delta_actual_max_abs"] = float(
                delta.abs().max().cpu()
            )
            stats["condition/cross_attention_v4/base_actual_rms"] = float(
                base_rms.cpu()
            )
            stats["condition/cross_attention_v4/delta_over_base_rms"] = float(
                (delta_rms / (base_rms + 1e-8)).cpu()
            )
            stats["condition/cross_attention_v4/mismatch_delta_actual_rms"] = float(
                mismatch_rms.cpu()
            )
            stats["condition/cross_attention_v4/mismatch_over_base_rms"] = float(
                (mismatch_rms / (base_rms + 1e-8)).cpu()
            )
            stats["condition/cross_attention_v4/aligned_over_mismatch_rms"] = float(
                (delta_rms / (mismatch_rms + 1e-8)).cpu()
            )
            stats["condition/cross_attention_v4/nodeform_delta_actual_max_abs"] = float(
                self.cross_attention_v4_nodeform_output.detach().abs().max().cpu()
            )

        if include_sensitivity:
            stats.update(self.compute_condition_sensitivity_stats())
        return stats

    def compute_condition_sensitivity_stats(self):
        if (
            not self.use_conditional_norm
            or self.motion_features is None
            or not hasattr(self, "_last_unet_uv_offsets")
        ):
            return {}

        with torch.no_grad():
            uv = self._last_unet_uv_offsets.detach()
            if uv.ndim != 4 or uv.shape[0] != 1:
                return {}

            device = uv.device
            dtype = uv.dtype
            current_feature = self._current_motion_feature(
                uv.shape[0],
                device,
                dtype,
            ).detach()
            n_motion = self.motion_features.shape[0]
            timestep = int(self.timestep) if self.timestep is not None else 0
            alt_idx = (timestep + max(1, n_motion // 2)) % n_motion
            alt_feature = self.motion_features[[alt_idx]].to(device=device, dtype=dtype)
            zero_feature = torch.zeros_like(current_feature)
            if hasattr(self, "_condition_monitor_prev_uv_offsets") and self._condition_monitor_prev_uv_offsets.shape == uv.shape:
                alt_uv = self._condition_monitor_prev_uv_offsets.to(device=device, dtype=dtype)
            else:
                alt_uv = torch.zeros_like(uv)

            base_deform, _ = self.forward_unet(
                uv,
                motion_feature_override=current_feature,
                save_debug=False,
            )
            base_delta = (
                self.spatial_residual_v2_deform_output.detach().clone()
                if self.spatial_residual_v2_deform_output is not None else None
            )
            alt_condition_deform, _ = self.forward_unet(
                uv,
                motion_feature_override=alt_feature,
                save_debug=False,
            )
            alt_condition_delta = (
                self.spatial_residual_v2_deform_output.detach().clone()
                if self.spatial_residual_v2_deform_output is not None else None
            )
            zero_condition_deform, _ = self.forward_unet(
                uv,
                motion_feature_override=zero_feature,
                save_debug=False,
            )
            zero_condition_delta = (
                self.spatial_residual_v2_deform_output.detach().clone()
                if self.spatial_residual_v2_deform_output is not None else None
            )
            alt_uv_deform, _ = self.forward_unet(
                alt_uv,
                motion_feature_override=current_feature,
                save_debug=False,
            )
            alt_uv_delta = (
                self.spatial_residual_v2_deform_output.detach().clone()
                if self.spatial_residual_v2_deform_output is not None else None
            )
            self._condition_monitor_prev_uv_offsets = uv.detach().clone()

            fixed_uv_alt_cond = self._tensor_diff_stats(base_deform, alt_condition_deform)
            fixed_uv_zero_cond = self._tensor_diff_stats(base_deform, zero_condition_deform)
            fixed_cond_alt_uv = self._tensor_diff_stats(base_deform, alt_uv_deform)
        stats = {
            "condition/sensitivity/fixed_uv_alt_condition_max_abs": fixed_uv_alt_cond["max_abs"],
            "condition/sensitivity/fixed_uv_alt_condition_mean_abs": fixed_uv_alt_cond["mean_abs"],
            "condition/sensitivity/fixed_uv_zero_condition_max_abs": fixed_uv_zero_cond["max_abs"],
            "condition/sensitivity/fixed_uv_zero_condition_mean_abs": fixed_uv_zero_cond["mean_abs"],
            "condition/sensitivity/fixed_condition_alt_uv_max_abs": fixed_cond_alt_uv["max_abs"],
            "condition/sensitivity/fixed_condition_alt_uv_mean_abs": fixed_cond_alt_uv["mean_abs"],
        }
        if (
            base_delta is not None
            and alt_condition_delta is not None
            and zero_condition_delta is not None
            and alt_uv_delta is not None
        ):
            delta_alt_cond = self._tensor_diff_stats(base_delta, alt_condition_delta)
            delta_zero_cond = self._tensor_diff_stats(base_delta, zero_condition_delta)
            delta_alt_uv = self._tensor_diff_stats(base_delta, alt_uv_delta)
            stats.update(
                {
                    "condition/spatial_residual_v2/sensitivity/fixed_uv_alt_condition_delta_max_abs": delta_alt_cond["max_abs"],
                    "condition/spatial_residual_v2/sensitivity/fixed_uv_alt_condition_delta_mean_abs": delta_alt_cond["mean_abs"],
                    "condition/spatial_residual_v2/sensitivity/fixed_uv_zero_condition_delta_max_abs": delta_zero_cond["max_abs"],
                    "condition/spatial_residual_v2/sensitivity/fixed_uv_zero_condition_delta_mean_abs": delta_zero_cond["mean_abs"],
                    "condition/spatial_residual_v2/sensitivity/fixed_condition_alt_uv_delta_max_abs": delta_alt_uv["max_abs"],
                    "condition/spatial_residual_v2/sensitivity/fixed_condition_alt_uv_delta_mean_abs": delta_alt_uv["mean_abs"],
                    "condition/spatial_residual_v2/sensitivity/delta_ratio_cond_over_uv": (
                        delta_alt_cond["mean_abs"] / (delta_alt_uv["mean_abs"] + 1e-8)
                    ),
                }
            )
        return stats
    
    def update_mesh_properties(self, verts, offsets):        
        remeshed_verts = self.uv_remesh_flame_vertices(verts)
        remeshed_verts = einops.rearrange(remeshed_verts, 'b h w c -> b (h w) c')
        remeshed_offsets = self.uv_remesh_flame_vertices(offsets) / STD_DEFORM
        remeshed_offsets = einops.rearrange(remeshed_offsets, 'b h w c -> b c h w')
        self._last_unet_uv_offsets = remeshed_offsets.detach()
        
        deform_output, nodeform_output = self.forward_unet(remeshed_offsets)
        remeshed_deform = einops.rearrange(deform_output, 'b c h w -> b (h w) c')
        nodeform_offsets = einops.rearrange(nodeform_output, 'b c h w -> b (h w) c')

        self.deform_output = deform_output
        self.neutral_output = nodeform_output

        verts = remeshed_verts + remeshed_deform
        faces = self.uv_remesh_faces

        nodeform_verts = remeshed_verts + nodeform_offsets

        triangles = verts[:, faces]
        nodeform_triangles = nodeform_verts[:, faces]

        # neutral gaussian deformations
        nodeform_face_center = nodeform_triangles.mean(dim=-2).squeeze(0)
        # compute undeformed face orientation and scale (no U-Net deformation)
        nodeform_face_orien_mat, nodeform_face_scaling = compute_face_orientation(
            nodeform_verts.squeeze(0), 
            faces.squeeze(0), 
            return_scale=True
        )
        self.neutral_face_orien_mat = nodeform_face_orien_mat
        self.xyz_neutral = self.compute_face_xyz_transformed(nodeform_face_center, nodeform_face_orien_mat, nodeform_face_scaling)

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # compute deformed face orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(
            verts.squeeze(0), 
            faces.squeeze(0), 
            return_scale=True
        )
        self.face_orien_quat = roma.quat_xyzw_to_wxyz(roma.rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces

    def _get_condition_residual(self):
        if not self.use_conditional_norm:
            return None

        if self.motion_condition_mode == "cross_attention_v3":
            condition_residual = self.cross_attention_v3_deform_output
        elif self.motion_condition_mode == "cross_attention_v4":
            condition_residual = self.cross_attention_v4_deform_output
        elif hasattr(self.deform_net, "get_condition_residual"):
            condition_residual = self.deform_net.get_condition_residual()
        else:
            condition_residual = None
        if condition_residual is None:
            return None
        if condition_residual.ndim != 4 or condition_residual.shape[1] != 3:
            raise ValueError(
                "Expected condition residual shape [B, 3, H, W], got "
                f"{tuple(condition_residual.shape)}"
            )
        if condition_residual.shape[0] == 2 and self.deform_output.shape[0] == 1:
            condition_residual = condition_residual[:1]
        if condition_residual.shape != self.deform_output.shape:
            raise ValueError(
                "Condition residual must match deformation output shape after selecting the driven half: "
                f"residual={tuple(condition_residual.shape)}, deform={tuple(self.deform_output.shape)}"
            )
        return condition_residual

    def compute_motion_residual_l2_loss(self):
        condition_residual = self._get_condition_residual()
        if condition_residual is None:
            return torch.tensor(0., device=self.deform_output.device)
        return (condition_residual ** 2).mean()

    def compute_motion_residual_laplacian_loss(self):
        condition_residual = self._get_condition_residual()
        if condition_residual is None:
            return torch.tensor(0., device=self.deform_output.device)

        kernel = torch.tensor(
            [[0., -1., 0.],
             [-1., 4., -1.],
             [0., -1., 0.]],
            device=condition_residual.device,
            dtype=condition_residual.dtype,
        ).view(1, 1, 3, 3)

        b_ = condition_residual.shape[0]
        residual = einops.rearrange(condition_residual, 'b c h w -> (b c) 1 h w')
        lap = F.conv2d(residual, kernel)
        lap = einops.rearrange(lap, '(b c) 1 h w -> b c h w', b=b_)
        return (lap ** 2).sum(dim=1, keepdim=True).mean()

    def compute_motion_residual_ratio_loss(self, ratio_limit):
        if ratio_limit < 0.0:
            raise ValueError("motion residual ratio limit must be non-negative.")
        condition_residual = self._get_condition_residual()
        if condition_residual is None or self.deform_base_output is None:
            zero = torch.tensor(0., device=self.deform_output.device)
            return zero, zero.detach()

        residual_rms = condition_residual.square().mean().clamp_min(1e-24).sqrt()
        base_rms = self.deform_base_output.detach().square().mean().clamp_min(1e-24).sqrt()
        residual_ratio = residual_rms / (base_rms + 1e-8)
        ratio_penalty = F.relu(residual_ratio - float(ratio_limit)).square()
        return ratio_penalty, residual_ratio

    def compute_motion_mismatch_suppression_loss(self):
        mismatch_residual = self.cross_attention_v4_mismatch_deform_output
        if mismatch_residual is None or self.deform_base_output is None:
            return torch.tensor(0., device=self.deform_output.device)
        base_energy = self.deform_base_output.detach().square().mean().clamp_min(1e-10)
        return mismatch_residual.square().mean() / base_energy
    
    def compute_laplacian_loss(self):
        kernel = torch.tensor(
            [[0., -1., 0.],
             [-1., 4., -1.],
             [0., -1., 0.]], device=self.deform_output.device,
        ).view(1, 1, 3, 3)

        b_ = self.deform_output.shape[0]
        deform = einops.rearrange(self.deform_output / STD_DEFORM, 'b c h w -> (b c) 1 h w')
        lap = F.conv2d(deform, kernel)
        lap = einops.rearrange(lap, '(b c) 1 h w -> b c h w', b=b_)
        lap = (lap ** 2).sum(dim=1, keepdim=True)
        self.laplacian = lap

        return lap.mean()

    def compute_neck_loss(self):
        if not self.static_neck:
            neck_rot_offset = self.neck_rot_offset(
                torch.tensor([self.timestep], dtype=torch.long, device=self.deform_output.device)
            )
            return neck_rot_offset.norm(dim=-1).mean()
        else:
            return 0.
    
    def print_neck_statistics(self):
        print(
            "mean:", self.neck_rot_offset.weight.mean(dim=0).detach(),
            "std:", self.neck_rot_offset.weight.std(dim=0).detach(),
        )
    
    def compute_relative_deformation_loss(self):
        # L2:
        diff = (((self.xyz_neutral - self.get_xyz) / STD_DEFORM) ** 2).sum(dim=1, keepdim=True)
        
        return diff.mean()
    
    def compute_relative_rotation_loss(self):
        # L2:
        relative_rot = self.neutral_face_orien_mat.inverse() @ self.face_orien_mat
            
        relative_rot = roma.rotmat_to_rotvec(relative_rot)

        diff = (relative_rot ** 2).sum(dim=-1)
        
        return diff.mean()
    
    def training_setup(self, training_args):
        super().training_setup(training_args)

        cross_attention_params = []
        cross_attention_param_ids = set()
        if (
            self.motion_condition_mode in (
                "cross_attention",
                "cross_attention_v2",
                "cross_attention_v3",
                "cross_attention_v4",
            )
            and self.motion_cross_attention_lr_mult > 0
        ):
            deform_net = self.deform_net.module if hasattr(self.deform_net, "module") else self.deform_net
            for module in deform_net.modules():
                if module.__class__.__name__ != "CrossAttentionCondition2d":
                    continue
                for param in module.parameters():
                    if not param.requires_grad:
                        continue
                    cross_attention_params.append(param)
                    cross_attention_param_ids.add(id(param))

        deform_params = [
            param
            for param in self.deform_net.parameters()
            if id(param) not in cross_attention_param_ids
        ]
        if self.motion_condition_proj is not None:
            deform_params += list(self.motion_condition_proj.parameters())

        # U-Net:
        if deform_params:
            self.optimizer.add_param_group(
                {
                    'params': deform_params,
                    'lr': training_args.deform_net_lr_init,
                    'weight_decay': training_args.deform_net_w_decay,
                    'name': "deform_net",
                }
            )
        if cross_attention_params:
            self.optimizer.add_param_group(
                {
                    'params': cross_attention_params,
                    'lr': training_args.deform_net_lr_init * self.motion_cross_attention_lr_mult,
                    'weight_decay': self.motion_cross_attention_w_decay,
                    'name': "deform_net_cross_attention",
                }
            )
            print(
                "Cross-attention optimizer group:",
                f"params={sum(param.numel() for param in cross_attention_params)}",
                f"lr_mult={self.motion_cross_attention_lr_mult}",
                f"weight_decay={self.motion_cross_attention_w_decay}",
            )

        if self.motion_cross_attention_adapter_only:
            for param_group in self.optimizer.param_groups:
                train_group = param_group["name"] == "deform_net_cross_attention"
                for param in param_group["params"]:
                    param.requires_grad_(train_group)
            print("Cross-attention adapter-only optimization: all non-attention parameters frozen.")

        self.deform_net_scheduler_args = get_expon_lr_func(
            lr_init=training_args.deform_net_lr_init,
            lr_final=training_args.deform_net_lr_final,
            lr_delay_mult=training_args.deform_net_lr_delay_mult,
            max_steps=training_args.deform_net_lr_max_steps,
        )

        if not self.static_neck:
            self.neck_rot_offset.requires_grad = True
            self.neck_optimizer = torch.optim.SparseAdam([{
                    'params': self.neck_rot_offset.parameters(),
                    'lr': training_args.neck_lr_init,
                    'name': "neck_rot_offset",
                }], 
                lr=training_args.neck_lr_init, 
                eps=1e-18,
            )
            self.neck_scheduler_args = get_expon_lr_func(
                lr_init=training_args.neck_lr_init,
                lr_final=training_args.neck_lr_final,
                lr_delay_mult=training_args.neck_lr_delay_mult,
                max_steps=training_args.neck_lr_max_steps,
            )
            if self.motion_cross_attention_adapter_only:
                for param in self.neck_rot_offset.parameters():
                    param.requires_grad_(False)

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

        if not self.static_neck:
            self.neck_optimizer.step()
            self.neck_optimizer.zero_grad(set_to_none = True)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.motion_condition_mode in (
            "cross_attention_v2",
            "cross_attention_v3",
            "cross_attention_v4",
        ):
            pretrain_iters = self.motion_cross_attention_base_pretrain_iters
            self._cross_attention_base_pretrain_active = iteration <= pretrain_iters
            if self._cross_attention_base_pretrain_active:
                self._cross_attention_base_lr_scale = 1.0
                self._cross_attention_condition_lr_scale = 0.0
            else:
                self._cross_attention_base_lr_scale = self.motion_cross_attention_base_lr_mult_after_pretrain
                warmup_iters = self.motion_cross_attention_condition_warmup_iters
                if warmup_iters > 0:
                    elapsed = max(0, iteration - pretrain_iters)
                    self._cross_attention_condition_lr_scale = min(1.0, elapsed / warmup_iters)
                else:
                    self._cross_attention_condition_lr_scale = 1.0
        else:
            self._cross_attention_base_pretrain_active = False
            self._cross_attention_base_lr_scale = 1.0
            self._cross_attention_condition_lr_scale = 1.0

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform_net":
                lr = self.deform_net_scheduler_args(iteration) * self._cross_attention_base_lr_scale
                param_group['lr'] = lr
            elif param_group["name"] == "deform_net_cross_attention":
                lr = (
                    self.deform_net_scheduler_args(iteration)
                    * self.motion_cross_attention_lr_mult
                    * self._cross_attention_condition_lr_scale
                )
                param_group['lr'] = lr
            
        if not self.static_neck:
            for param_group in self.neck_optimizer.param_groups:
                if param_group["name"] == "neck_rot_offset":
                    lr = self.neck_scheduler_args(iteration)
                    param_group['lr'] = lr

        super().update_learning_rate(iteration)

        if self.motion_cross_attention_adapter_only:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] != "deform_net_cross_attention":
                    param_group["lr"] = 0.0
            if not self.static_neck:
                for param_group in self.neck_optimizer.param_groups:
                    param_group["lr"] = 0.0

    def capture(self):
        # save flame shape and base rotation for reenactment
        return {
            "shape": self.flame_param["shape"],
            "base_rot": self.flame_param["base_rot"],
            "deform_net": self.deform_net.state_dict(),
            "motion_condition_mode": self.motion_condition_mode,
            "motion_condition_runtime_shuffle": self.motion_condition_runtime_shuffle,
            "motion_nodeform_condition": self.motion_nodeform_condition,
            "motion_feature_center": self.motion_feature_center.detach().cpu(),
            "motion_feature_center_source": self.motion_feature_center_source,
            "motion_feature_common_energy_ratio": self.motion_feature_common_energy_ratio,
            "motion_feature_centered_rms": self.motion_feature_centered_rms,
            "motion_condition_proj": (
                self.motion_condition_proj.state_dict()
                if self.motion_condition_proj is not None else None
            ),
            "gaussians": super().capture(),
        }

    @staticmethod
    def _legacy_unet_key_to_conditional(key):
        """Map the original sequential recursive U-Net keys to conditional blocks."""
        if key.startswith("module."):
            key = key[len("module."):]
        prefix = "model.model."
        if not key.startswith(prefix):
            return key

        tokens = key[len(prefix):].split(".")
        mapped = ["model"]
        offset = 0
        outermost = True
        while offset < len(tokens):
            layer_index = tokens[offset]
            offset += 1
            if offset >= len(tokens):
                return None

            if tokens[offset] == "model":
                expected_submodule_index = "1" if outermost else "3"
                if layer_index != expected_submodule_index:
                    return None
                mapped.append("submodule")
                offset += 1
                outermost = False
                continue

            if outermost:
                leaf_map = {
                    "0": ("down", "0"),
                    "3": ("up", "1"),
                }
            else:
                leaf_map = {
                    "1": ("down", "1"),
                    "2": ("down", "2"),
                    "3": ("up", "1"),
                    "4": ("up", "2"),
                    "5": ("up", "1"),
                    "6": ("up", "2"),
                }
            if layer_index not in leaf_map:
                return None
            mapped.extend(leaf_map[layer_index])
            mapped.extend(tokens[offset:])
            return ".".join(mapped)
        return None

    def restore(self, chkpt, training_args=None):
        self.flame_param["shape"] = chkpt["shape"]
        self.flame_param["base_rot"] = chkpt["base_rot"]
        chkpt_motion_mode = chkpt.get("motion_condition_mode", None)
        if chkpt_motion_mode is not None and chkpt_motion_mode != self.motion_condition_mode:
            print(
                "WARNING: checkpoint motion_condition_mode differs from current config:",
                f"checkpoint={chkpt_motion_mode}",
                f"current={self.motion_condition_mode}",
            )
        try:
            self.deform_net.load_state_dict(chkpt["deform_net"])
        except RuntimeError as error:
            if not self.use_conditional_norm:
                raise
            current_state = self.deform_net.state_dict()
            skip_checkpoint_attention = (
                self.motion_condition_mode in (
                    "cross_attention_v2",
                    "cross_attention_v3",
                    "cross_attention_v4",
                )
                and chkpt_motion_mode != self.motion_condition_mode
            )
            compatible_state = {}
            translated_legacy_keys = []
            for source_key, value in chkpt["deform_net"].items():
                target_key = source_key
                if (
                    target_key not in current_state
                    and self.motion_condition_mode in (
                        "cross_attention_v3",
                        "cross_attention_v4",
                    )
                ):
                    target_key = self._legacy_unet_key_to_conditional(source_key)
                if target_key is None or target_key not in current_state:
                    continue
                if current_state[target_key].shape != value.shape:
                    continue
                if skip_checkpoint_attention and ".cross_attention_condition." in source_key:
                    continue
                compatible_state[target_key] = value
                if target_key != source_key:
                    translated_legacy_keys.append((source_key, target_key))
            skipped_attention_keys = [
                key
                for key in chkpt["deform_net"]
                if skip_checkpoint_attention and ".cross_attention_condition." in key
            ]
            skipped_shape_keys = [
                key
                for key, value in chkpt["deform_net"].items()
                if key in current_state and current_state[key].shape != value.shape
            ]
            if (
                self.motion_cross_attention_adapter_only
                and self.motion_condition_mode in (
                    "cross_attention_v3",
                    "cross_attention_v4",
                )
            ):
                required_base_keys = {"model.down.0.weight", "model.up.1.weight"}
                missing_base_keys = required_base_keys.difference(compatible_state)
                if missing_base_keys:
                    raise RuntimeError(
                        "Adapter-only initialization refused to freeze an unloaded deformation U-Net. "
                        f"Missing migrated baseline keys: {sorted(missing_base_keys)}"
                    ) from error
            incompatible = self.deform_net.load_state_dict(compatible_state, strict=False)
            print(
                "Loaded deformation U-Net with compatible-only state dict for conditional-norm mode:",
                f"source_keys={len(chkpt['deform_net'])}",
                f"loaded_keys={len(compatible_state)}",
                f"translated_legacy={len(translated_legacy_keys)}",
                f"skipped_attention={len(skipped_attention_keys)}",
                f"shape_mismatch={len(skipped_shape_keys)}",
                f"missing={len(incompatible.missing_keys)}",
                f"unexpected={len(incompatible.unexpected_keys)}",
            )
            print("Original strict-load error:", str(error).splitlines()[0])
            if skipped_shape_keys:
                print("Skipped shape-mismatched keys:", skipped_shape_keys[:20])
            if skipped_attention_keys:
                print("Skipped checkpoint attention keys:", skipped_attention_keys[:20])
            if incompatible.missing_keys:
                print("New/randomly initialized keys:", incompatible.missing_keys[:20])
            if incompatible.unexpected_keys:
                print("Unused checkpoint keys:", incompatible.unexpected_keys[:20])

        if self.motion_condition_proj is not None and chkpt.get("motion_condition_proj") is not None:
            self.motion_condition_proj.load_state_dict(chkpt["motion_condition_proj"])
        elif self.motion_condition_proj is not None:
            print("WARNING: legacy motion_condition_proj is enabled but not present in checkpoint.")
        if (
            self.motion_condition_mode == "cross_attention_v4"
            and chkpt.get("motion_feature_center") is not None
        ):
            center = torch.as_tensor(chkpt["motion_feature_center"]).detach().float()
            if center.shape != (1, 512):
                raise ValueError(
                    "Checkpoint motion_feature_center must have shape [1, 512], got "
                    f"{tuple(center.shape)}"
                )
            self.motion_feature_center = center.cpu()
            self.motion_feature_center_source = str(
                chkpt.get("motion_feature_center_source", "checkpoint")
            )
            self.motion_feature_common_energy_ratio = float(
                chkpt.get(
                    "motion_feature_common_energy_ratio",
                    self.motion_feature_common_energy_ratio,
                )
            )
            self.motion_feature_centered_rms = float(
                chkpt.get(
                    "motion_feature_centered_rms",
                    self.motion_feature_centered_rms,
                )
            )
            self._motion_feature_center_restored = True
            print(
                "Restored Xnemo training feature center:",
                f"source={self.motion_feature_center_source}",
                f"common_energy_ratio={self.motion_feature_common_energy_ratio:.6f}",
                f"centered_rms={self.motion_feature_centered_rms:.6f}",
            )
        super().restore(chkpt["gaussians"], training_args)
