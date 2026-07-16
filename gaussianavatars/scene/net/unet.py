import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools


###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return nn.Sequential()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class ConditionalAffine2d(nn.Module):
    """FiLM-style conditional affine modulation for 2D features.

    With norm_type="existing", this preserves the incoming U-Net normalization and only
    applies condition-generated scale/bias. This is a FiLM baseline, not strict AdaIN.
    """

    def __init__(
        self,
        num_features,
        condition_dim=512,
        hidden_dim=128,
        gamma_scale=0.1,
        norm_type="existing",
    ):
        super().__init__()
        self.num_features = num_features
        self.condition_dim = condition_dim
        self.gamma_scale = gamma_scale

        if norm_type in ("existing", "none"):
            self.norm = nn.Identity()
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        elif norm_type == "group":
            num_groups = 32 if num_features % 32 == 0 else 1
            self.norm = nn.GroupNorm(num_groups, num_features, affine=False)
        else:
            raise NotImplementedError('conditional norm [%s] is not found' % norm_type)

        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_features * 2),
        )
        self.reset_to_identity()

    def reset_to_identity(self):
        last = self.mlp[-1]
        init.constant_(last.weight.data, 0.0)
        init.constant_(last.bias.data, 0.0)

    def forward(self, x, condition):
        if condition is None:
            raise ValueError("ConditionalAffine2d requires a condition tensor.")
        if condition.ndim != 2:
            raise ValueError(
                f"Expected condition shape [B, {self.condition_dim}], got {tuple(condition.shape)}"
            )
        if condition.shape[0] != x.shape[0]:
            raise ValueError(
                f"Condition batch {condition.shape[0]} must match feature batch {x.shape[0]}"
            )
        if condition.shape[1] != self.condition_dim:
            raise ValueError(
                f"Expected condition dim {self.condition_dim}, got {condition.shape[1]}"
            )

        gamma, beta = self.mlp(condition).chunk(2, dim=1)
        gamma = gamma * self.gamma_scale
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return (1.0 + gamma) * self.norm(x) + beta


class ConditionalInstanceNorm2d(nn.Module):
    """Strict AdaIN / conditional instance normalization for [B, C, H, W] features."""

    def __init__(
        self,
        num_features,
        condition_dim=512,
        hidden_dim=128,
        gamma_scale=0.1,
        eps=1e-5,
    ):
        super().__init__()
        self.num_features = num_features
        self.condition_dim = condition_dim
        self.gamma_scale = gamma_scale
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_features * 2),
        )
        self.reset_to_identity()

    def reset_to_identity(self):
        last = self.mlp[-1]
        init.constant_(last.weight.data, 0.0)
        init.constant_(last.bias.data, 0.0)

    def forward(self, x, condition):
        if x.ndim != 4:
            raise ValueError(f"ConditionalInstanceNorm2d expects [B, C, H, W], got {tuple(x.shape)}")
        if condition is None:
            raise ValueError("ConditionalInstanceNorm2d requires a condition tensor.")
        if condition.ndim != 2:
            raise ValueError(
                f"Expected condition shape [B, {self.condition_dim}], got {tuple(condition.shape)}"
            )
        if condition.shape[0] != x.shape[0]:
            raise ValueError(
                f"Condition batch {condition.shape[0]} must match feature batch {x.shape[0]}"
            )
        if condition.shape[1] != self.condition_dim:
            raise ValueError(
                f"Expected condition dim {self.condition_dim}, got {condition.shape[1]}"
            )
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected feature channels {self.num_features}, got {x.shape[1]}"
            )

        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        x_norm = (x - mean) * torch.rsqrt(var + self.eps)
        gamma, beta = self.mlp(condition).chunk(2, dim=1)
        gamma = gamma * self.gamma_scale
        return (1.0 + gamma[:, :, None, None]) * x_norm + beta[:, :, None, None]


class SharedConditionTrunk(nn.Module):
    """Shared 512-dim condition trunk for all strict_adain_allnorm sites."""

    def __init__(self, condition_dim=512, hidden_dim=128):
        super().__init__()
        self.condition_dim = condition_dim
        self.output_dim = hidden_dim
        self.net = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, condition):
        if condition is None:
            raise ValueError("SharedConditionTrunk requires a condition tensor.")
        if condition.ndim != 2:
            raise ValueError(
                f"Expected condition shape [B, {self.condition_dim}], got {tuple(condition.shape)}"
            )
        if condition.shape[1] != self.condition_dim:
            raise ValueError(
                f"Expected condition dim {self.condition_dim}, got {condition.shape[1]}"
            )
        return self.net(condition)


class SharedStrictAdaIN2d(nn.Module):
    """Strict AdaIN slot with a shared condition trunk and a site-specific head."""

    def __init__(
        self,
        num_features,
        condition_trunk,
        gamma_scale=0.1,
        eps=1e-5,
        site_name="adain",
        head_init_std=1e-4,
    ):
        super().__init__()
        self.num_features = num_features
        self.condition_trunk = condition_trunk
        self.gamma_scale = gamma_scale
        self.eps = eps
        self.site_name = site_name
        self.head_init_std = head_init_std
        self.head = nn.Linear(condition_trunk.output_dim, num_features * 2)
        self.last_stats = {}
        self.reset_to_identity()

    def reset_to_identity(self):
        init.normal_(self.head.weight.data, 0.0, self.head_init_std)
        init.constant_(self.head.bias.data, 0.0)

    def forward(self, x, condition):
        if x.ndim != 4:
            raise ValueError(f"SharedStrictAdaIN2d expects [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected feature channels {self.num_features}, got {x.shape[1]}"
            )
        if condition.shape[0] != x.shape[0]:
            raise ValueError(
                f"Condition batch {condition.shape[0]} must match feature batch {x.shape[0]}"
            )

        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        x_norm = (x - mean) * torch.rsqrt(var + self.eps)
        gamma, beta = self.head(self.condition_trunk(condition)).chunk(2, dim=1)
        gamma = gamma * self.gamma_scale
        y = (1.0 + gamma[:, :, None, None]) * x_norm + beta[:, :, None, None]

        with torch.no_grad():
            delta = y - x_norm
            self.last_stats = {
                "site": self.site_name,
                "channels": int(self.num_features),
                "gamma_mean": float(gamma.detach().mean().cpu()),
                "gamma_std": float(gamma.detach().std(unbiased=False).cpu()),
                "beta_mean": float(beta.detach().mean().cpu()),
                "beta_std": float(beta.detach().std(unbiased=False).cpu()),
                "feature_delta_mean_abs": float(delta.detach().abs().mean().cpu()),
                "feature_delta_max_abs": float(delta.detach().abs().max().cpu()),
            }
        return y


class CrossAttentionCondition2d(nn.Module):
    """Spatial cross-attention from U-Net features to Xnemo condition tokens.

    Each spatial feature vector is a query. A single 512-dim frame condition is
    expanded into multiple learned tokens that provide keys and values. The
    module returns a small feature residual, not a deformation-map residual.
    """

    def __init__(
        self,
        num_features,
        condition_dim=512,
        hidden_dim=128,
        num_tokens=8,
        attention_dim=64,
        num_heads=4,
        residual_gate_init=0.05,
        site_name="cross_attention",
        output_init_std=2e-2,
        logit_scale_init=4.0,
        normalize_qk=True,
        direct_token_projection=False,
        use_spatial_position=False,
    ):
        super().__init__()
        if num_tokens < 2:
            raise ValueError("cross_attention requires at least two condition tokens.")
        attention_dim = max(8, int(attention_dim))
        num_heads = max(1, min(int(num_heads), attention_dim))
        while attention_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1

        residual_gate_init = min(max(float(residual_gate_init), 1e-4), 1. - 1e-4)
        output_init_std = max(float(output_init_std), 0.0)
        logit_scale_init = max(float(logit_scale_init), 1e-3)
        self.num_features = int(num_features)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_tokens = int(num_tokens)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attention_dim // self.num_heads
        self.site_name = site_name
        self.output_init_std = output_init_std
        self.normalize_qk = bool(normalize_qk)
        self.direct_token_projection = bool(direct_token_projection)
        self.use_spatial_position = bool(use_spatial_position)

        num_groups = 32 if self.num_features % 32 == 0 else 1
        self.feature_norm = nn.GroupNorm(num_groups, self.num_features, affine=False)
        if self.direct_token_projection:
            self.condition_to_tokens = nn.Sequential(
                nn.LayerNorm(self.condition_dim),
                nn.Linear(self.condition_dim, self.num_tokens * self.hidden_dim),
            )
        else:
            self.condition_to_tokens = nn.Sequential(
                nn.LayerNorm(self.condition_dim),
                nn.Linear(self.condition_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.num_tokens * self.hidden_dim),
            )
        self.query_proj = nn.Conv2d(self.num_features, self.attention_dim, kernel_size=1)
        self.position_proj = (
            nn.Linear(2, self.attention_dim, bias=False)
            if self.use_spatial_position else None
        )
        self.key_proj = nn.Linear(self.hidden_dim, self.attention_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.attention_dim)
        self.out_proj = nn.Conv2d(self.attention_dim, self.num_features, kernel_size=1)
        self.gate_logit = nn.Parameter(torch.logit(torch.tensor(residual_gate_init)))
        self.logit_scale_log = nn.Parameter(torch.log(torch.tensor(float(logit_scale_init))))
        self.key_token_bias = nn.Parameter(torch.empty(self.num_tokens, self.hidden_dim))
        self.last_stats = {}
        self.last_active_entropy_ratio = None
        self.last_token_balance_loss = None
        self.reset_to_identity()

    def reset_to_identity(self):
        init.normal_(self.key_token_bias.data, 0.0, 0.02)
        init.normal_(self.out_proj.weight.data, 0.0, self.output_init_std)
        if self.out_proj.bias is not None:
            init.constant_(self.out_proj.bias.data, 0.0)

    def forward(self, x, condition):
        if x.ndim != 4:
            raise ValueError(f"CrossAttentionCondition2d expects [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected feature channels {self.num_features}, got {x.shape[1]}"
            )
        if condition is None:
            raise ValueError("CrossAttentionCondition2d requires a condition tensor.")
        if condition.ndim != 2 or condition.shape[1] != self.condition_dim:
            raise ValueError(
                f"Expected condition shape [B, {self.condition_dim}], got {tuple(condition.shape)}"
            )
        if condition.shape[0] != x.shape[0]:
            raise ValueError(
                f"Condition batch {condition.shape[0]} must match feature batch {x.shape[0]}"
            )

        b, _, h, w = x.shape
        tokens = self.condition_to_tokens(condition).view(b, self.num_tokens, self.hidden_dim)
        key_tokens = tokens + self.key_token_bias[None].to(device=tokens.device, dtype=tokens.dtype)

        q = self.query_proj(self.feature_norm(x)).flatten(2).transpose(1, 2)
        if self.position_proj is not None:
            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
                torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
                indexing="ij",
            )
            coords = torch.stack([xx, yy], dim=-1).view(h * w, 2)
            q = q + self.position_proj(coords)[None]
        q = q.view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(key_tokens).view(b, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(tokens).view(b, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        if self.normalize_qk:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            logit_scale = self.logit_scale_log.exp().clamp(0.1, 32.0).to(device=x.device, dtype=x.dtype)
        else:
            logit_scale = torch.tensor(self.head_dim ** -0.5, device=x.device, dtype=x.dtype)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * logit_scale
        attn = torch.softmax(attn_logits, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(b, h * w, self.attention_dim)
        context = context.transpose(1, 2).view(b, self.attention_dim, h, w)

        delta = self.out_proj(context)
        condition_mask = (condition.abs().sum(dim=1, keepdim=True) > 0).to(
            device=x.device,
            dtype=x.dtype,
        )
        active_samples = condition_mask[:, 0].bool()
        if bool(active_samples.any()):
            active_attn = attn[active_samples]
            active_entropy = -(
                active_attn * (active_attn + 1e-8).log()
            ).sum(dim=-1)
            max_entropy = torch.log(
                torch.tensor(float(self.num_tokens), device=x.device, dtype=x.dtype)
            )
            self.last_active_entropy_ratio = active_entropy.mean() / max_entropy
            token_usage = active_attn.mean(dim=(0, 1, 2))
            self.last_token_balance_loss = (
                token_usage * float(self.num_tokens) - 1.0
            ).square().mean()
        else:
            graph_zero = attn.sum() * 0.0
            self.last_active_entropy_ratio = graph_zero
            self.last_token_balance_loss = graph_zero
        gate = torch.sigmoid(self.gate_logit).to(device=x.device, dtype=x.dtype)
        y = x + gate * delta * condition_mask[:, :, None, None]

        with torch.no_grad():
            eps = 1e-8
            entropy = -(attn.detach() * (attn.detach() + eps).log()).sum(dim=-1)
            self.last_stats = {
                "site": self.site_name,
                "channels": int(self.num_features),
                "num_tokens": int(self.num_tokens),
                "hidden_dim": int(self.hidden_dim),
                "attention_dim": int(self.attention_dim),
                "num_heads": int(self.num_heads),
                "gate": float(gate.detach().cpu()),
                "logit_scale": float(logit_scale.detach().cpu()),
                "normalize_qk": float(self.normalize_qk),
                "direct_token_projection": float(self.direct_token_projection),
                "use_spatial_position": float(self.use_spatial_position),
                "key_token_bias_std": float(self.key_token_bias.detach().std(unbiased=False).cpu()),
                "token_mean": float(tokens.detach().mean().cpu()),
                "token_std": float(tokens.detach().std(unbiased=False).cpu()),
                "attention_entropy_mean": float(entropy.mean().cpu()),
                "attention_entropy_std": float(entropy.std(unbiased=False).cpu()),
                "attention_max_mean": float(attn.detach().max(dim=-1)[0].mean().cpu()),
                "active_condition_fraction": float(condition_mask.detach().mean().cpu()),
                "active_attention_entropy_ratio": float(
                    self.last_active_entropy_ratio.detach().cpu()
                ),
                "active_token_balance_loss": float(
                    self.last_token_balance_loss.detach().cpu()
                ),
                "feature_delta_mean_abs": float((gate * delta).detach().abs().mean().cpu()),
                "feature_delta_max_abs": float((gate * delta).detach().abs().max().cpu()),
            }
        return y


class ConditionalLayerNorm(nn.Module):
    """Conditional LayerNorm for token features laid out as [B, N, C]."""

    def __init__(
        self,
        num_features,
        condition_dim=512,
        hidden_dim=128,
        gamma_scale=0.1,
        eps=1e-5,
    ):
        super().__init__()
        self.num_features = num_features
        self.condition_dim = condition_dim
        self.gamma_scale = gamma_scale
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_features * 2),
        )
        self.reset_to_identity()

    def reset_to_identity(self):
        last = self.mlp[-1]
        init.constant_(last.weight.data, 0.0)
        init.constant_(last.bias.data, 0.0)

    def forward(self, x, condition):
        if x.ndim != 3:
            raise ValueError(f"ConditionalLayerNorm expects [B, N, C], got {tuple(x.shape)}")
        if condition is None:
            raise ValueError("ConditionalLayerNorm requires a condition tensor.")
        if condition.ndim != 2:
            raise ValueError(
                f"Expected condition shape [B, {self.condition_dim}], got {tuple(condition.shape)}"
            )
        if condition.shape[0] != x.shape[0]:
            raise ValueError(
                f"Condition batch {condition.shape[0]} must match feature batch {x.shape[0]}"
            )
        if condition.shape[1] != self.condition_dim:
            raise ValueError(
                f"Expected condition dim {self.condition_dim}, got {condition.shape[1]}"
            )
        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"Expected feature dim {self.num_features}, got {x.shape[-1]}"
            )

        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) * torch.rsqrt(var + self.eps)
        gamma, beta = self.mlp(condition).chunk(2, dim=1)
        gamma = gamma * self.gamma_scale
        return (1.0 + gamma[:, None, :]) * x_norm + beta[:, None, :]


class GatedConditionalAffine2d(nn.Module):
    """Residual gated wrapper around condition-generated affine modulation."""

    def __init__(
        self,
        num_features,
        condition_dim=512,
        hidden_dim=128,
        gamma_scale=0.1,
        norm_type="existing",
        gate_init=0.05,
    ):
        super().__init__()
        gate_init = min(max(float(gate_init), 1e-4), 1. - 1e-4)
        self.affine = ConditionalAffine2d(
            num_features,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            gamma_scale=gamma_scale,
            norm_type=norm_type,
        )
        self.gate_logit = nn.Parameter(torch.logit(torch.tensor(gate_init)))

    def reset_to_identity(self):
        self.affine.reset_to_identity()

    def forward(self, x, condition):
        gate = torch.sigmoid(self.gate_logit).to(device=x.device, dtype=x.dtype)
        return x + gate * (self.affine(x, condition) - x)


class ConditionalResidualBranch(nn.Module):
    """Predict a condition-dependent deformation residual from spatial U-Net features."""

    def __init__(
        self,
        feature_channels,
        output_channels,
        condition_dim=512,
        hidden_dim=128,
        gamma_scale=0.1,
        norm_type="existing",
        alpha_init=0.05,
    ):
        super().__init__()
        alpha_init = min(max(float(alpha_init), 1e-4), 1. - 1e-4)
        mid_channels = max(output_channels * 8, feature_channels // 2)
        self.condition_affine = ConditionalAffine2d(
            feature_channels,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            gamma_scale=gamma_scale,
            norm_type=norm_type,
        )
        self.head = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(feature_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(mid_channels, output_channels, kernel_size=4, stride=2, padding=1),
        )
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(alpha_init)))
        self.reset_to_identity()

    def reset_to_identity(self):
        self.condition_affine.reset_to_identity()
        final = self.head[-1]
        init.normal_(final.weight.data, 0.0, 1e-4)
        if final.bias is not None:
            init.constant_(final.bias.data, 0.0)

    def forward(self, spatial_feature, condition):
        conditioned_feature = self.condition_affine(spatial_feature, condition)
        residual = self.head(conditioned_feature)
        condition_mask = (condition.abs().sum(dim=1, keepdim=True) > 0).to(
            device=residual.device,
            dtype=residual.dtype,
        )
        residual = residual * condition_mask[:, :, None, None]
        alpha = torch.sigmoid(self.alpha_logit).to(device=residual.device, dtype=residual.dtype)
        return alpha * residual


class SpatialResidualScaleBlock(nn.Module):
    """One spatial scale of spatial_residual_branch_v2."""

    def __init__(
        self,
        feature_channels,
        input_channels,
        branch_channels,
        trunk_dim,
        gamma_scale=0.1,
        scale_name="scale",
        head_init_std=1e-4,
    ):
        super().__init__()
        self.scale_name = scale_name
        self.gamma_scale = gamma_scale
        self.head_init_std = head_init_std
        self.input_proj = nn.Conv2d(feature_channels + input_channels, branch_channels, kernel_size=1)
        self.input_norm = nn.InstanceNorm2d(branch_channels, affine=False, track_running_stats=False)
        self.refine = nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=1, padding=1)
        self.refine_norm = nn.InstanceNorm2d(branch_channels, affine=False, track_running_stats=False)
        self.act = nn.SiLU()
        self.condition_head = nn.Linear(trunk_dim, branch_channels * 2)
        self.last_stats = {}
        self.reset_to_identity()

    def reset_to_identity(self):
        init.normal_(self.condition_head.weight.data, 0.0, self.head_init_std)
        init.constant_(self.condition_head.bias.data, 0.0)

    def spatial_parameters(self):
        yield from self.input_proj.parameters()
        yield from self.refine.parameters()

    def forward(self, feature, unet_input, condition_code):
        if feature.ndim != 4 or unet_input.ndim != 4:
            raise ValueError("SpatialResidualScaleBlock expects 4D feature and input tensors.")
        spatial_input = F.interpolate(
            unet_input,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat([feature, spatial_input], dim=1)
        x = self.act(self.input_norm(self.input_proj(x)))
        gamma, beta = self.condition_head(condition_code).chunk(2, dim=1)
        gamma = gamma * self.gamma_scale
        x = (1.0 + gamma[:, :, None, None]) * x + beta[:, :, None, None]
        x = self.act(self.refine_norm(self.refine(x)))
        with torch.no_grad():
            self.last_stats = {
                "gamma_mean": float(gamma.detach().mean().cpu()),
                "gamma_std": float(gamma.detach().std(unbiased=False).cpu()),
                "beta_mean": float(beta.detach().mean().cpu()),
                "beta_std": float(beta.detach().std(unbiased=False).cpu()),
                "feature_mean_abs": float(x.detach().abs().mean().cpu()),
            }
        return x


class SpatialResidualBranchV2(nn.Module):
    """Predict a constrained residual deformation from real decoder features, UV/pos input, and 512 condition."""

    def __init__(
        self,
        input_channels,
        output_channels,
        scale_1x_channels,
        scale_2x_channels,
        condition_dim=512,
        hidden_dim=128,
        branch_channels=64,
        gamma_scale=0.1,
        final_init_std=1e-4,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.branch_channels = branch_channels
        self.final_init_std = final_init_std
        self.condition_trunk = SharedConditionTrunk(condition_dim=condition_dim, hidden_dim=hidden_dim)
        self.scale_2x = SpatialResidualScaleBlock(
            scale_2x_channels,
            input_channels,
            branch_channels,
            self.condition_trunk.output_dim,
            gamma_scale=gamma_scale,
            scale_name="scale_2x",
        )
        self.scale_1x = SpatialResidualScaleBlock(
            scale_1x_channels,
            input_channels,
            branch_channels,
            self.condition_trunk.output_dim,
            gamma_scale=gamma_scale,
            scale_name="scale_1x",
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_channels * 2, branch_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(branch_channels, output_channels, kernel_size=3, stride=1, padding=1),
        )
        self.last_stats = {}
        self.reset_to_identity()

    def reset_to_identity(self):
        self.scale_2x.reset_to_identity()
        self.scale_1x.reset_to_identity()
        final = self.fusion[-1]
        init.normal_(final.weight.data, 0.0, self.final_init_std)
        if final.bias is not None:
            init.constant_(final.bias.data, 0.0)

    @staticmethod
    def _grad_norm(parameters):
        total = 0.0
        for param in parameters:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            total += float((grad * grad).sum().cpu())
        return total ** 0.5

    def collect_grad_stats(self, prefix="condition/spatial_residual_v2"):
        return {
            f"{prefix}/shared_trunk_grad_norm": self._grad_norm(self.condition_trunk.parameters()),
            f"{prefix}/scale_2x_condition_head_grad_norm": self._grad_norm(self.scale_2x.condition_head.parameters()),
            f"{prefix}/scale_1x_condition_head_grad_norm": self._grad_norm(self.scale_1x.condition_head.parameters()),
            f"{prefix}/scale_2x_spatial_conv_grad_norm": self._grad_norm(self.scale_2x.spatial_parameters()),
            f"{prefix}/scale_1x_spatial_conv_grad_norm": self._grad_norm(self.scale_1x.spatial_parameters()),
            f"{prefix}/fusion_grad_norm": self._grad_norm(self.fusion.parameters()),
            f"{prefix}/output_head_grad_norm": self._grad_norm(self.fusion[-1].parameters()),
        }

    def forward(self, feature_collector, unet_input, condition):
        if condition is None:
            raise ValueError("SpatialResidualBranchV2 requires a condition tensor.")
        if unet_input.ndim != 4:
            raise ValueError(f"Expected unet_input [B,C,H,W], got {tuple(unet_input.shape)}")
        if unet_input.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected unet_input channels {self.input_channels}, got {unet_input.shape[1]}"
            )
        if condition.shape[0] != unet_input.shape[0]:
            raise ValueError(
                f"Condition batch {condition.shape[0]} must match input batch {unet_input.shape[0]}"
            )
        try:
            feature_2x = feature_collector["scale_2x"]
            feature_1x = feature_collector["scale_1x"]
        except KeyError as exc:
            raise KeyError(
                "spatial_residual_branch_v2 requires scale_2x and scale_1x decoder features."
            ) from exc

        condition_code = self.condition_trunk(condition)
        branch_2x = self.scale_2x(feature_2x, unet_input, condition_code)
        branch_1x = self.scale_1x(feature_1x, unet_input, condition_code)
        target_size = unet_input.shape[-2:]
        branch_2x = F.interpolate(branch_2x, size=target_size, mode="bilinear", align_corners=False)
        branch_1x = F.interpolate(branch_1x, size=target_size, mode="bilinear", align_corners=False)
        delta = self.fusion(torch.cat([branch_2x, branch_1x], dim=1))

        with torch.no_grad():
            flat = delta.detach().flatten(2)
            self.last_stats = {
                "delta_norm_mean": float(delta.detach().mean().cpu()),
                "delta_norm_mean_abs": float(delta.detach().abs().mean().cpu()),
                "delta_norm_std": float(delta.detach().std(unbiased=False).cpu()),
                "delta_norm_max_abs": float(delta.detach().abs().max().cpu()),
                "delta_norm_spatial_std": float(flat.std(dim=-1, unbiased=False).mean().cpu()),
            }
            for name, block in (("scale_2x", self.scale_2x), ("scale_1x", self.scale_1x)):
                for key, value in block.last_stats.items():
                    self.last_stats[f"{name}_{key}"] = value
        return delta


def define_G(
    input_nc,
    output_nc,
    ngf,
    netG,
    norm='batch',
    n_layers=None,
    use_dropout=False,
    init_type='normal',
    init_gain=0.02,
    gpu_ids=[],
    condition_mode='legacy_concat',
    condition_dim=512,
    condition_layers='bottleneck',
    condition_hidden_dim=128,
    condition_gamma_scale=0.1,
    condition_norm='existing',
    condition_gate_init=0.05,
    condition_residual_alpha_init=0.05,
    condition_num_tokens=8,
    condition_attention_dim=64,
    condition_cross_attention_gate_init=None,
    condition_attention_output_init_std=2e-2,
    condition_attention_logit_scale=4.0,
    condition_attention_direct_tokens=False,
    condition_attention_use_position=False,
):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    use_conditional_unet = condition_mode in (
        "adain",
        "strict_adain",
        "strict_adain_bottleneck",
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
        "cross_attention_v5",
    )
    generator_cls = ConditionalUnetGenerator if use_conditional_unet else UnetGenerator

    if netG == 'unet_64':
        num_downs = 5 if n_layers is None else n_layers
    elif netG == 'unet_128':
        num_downs = 7 if n_layers is None else n_layers
    elif netG == 'unet_256':
        num_downs = 8 if n_layers is None else n_layers
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    if use_conditional_unet:
        net = generator_cls(
            input_nc,
            output_nc,
            num_downs,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            condition_dim=condition_dim,
            condition_mode=condition_mode,
            condition_layers=condition_layers,
            condition_hidden_dim=condition_hidden_dim,
            condition_gamma_scale=condition_gamma_scale,
            condition_norm=condition_norm,
            condition_gate_init=condition_gate_init,
            condition_residual_alpha_init=condition_residual_alpha_init,
            condition_num_tokens=condition_num_tokens,
            condition_attention_dim=condition_attention_dim,
            condition_cross_attention_gate_init=condition_cross_attention_gate_init,
            condition_attention_output_init_std=condition_attention_output_init_std,
            condition_attention_logit_scale=condition_attention_logit_scale,
            condition_attention_direct_tokens=condition_attention_direct_tokens,
            condition_attention_use_position=condition_attention_use_position,
        )
    else:
        net = generator_cls(input_nc, output_nc, num_downs, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    net = init_net(net, init_type, init_gain, gpu_ids)
    if use_conditional_unet:
        target_net = net.module if isinstance(net, torch.nn.DataParallel) else net
        target_net.reset_condition_to_identity()
    return net


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

    def zero_last_layer(self):
        self.model.model[-1].weight.data *= 0
        self.model.model[-1].bias.data *= 0


class ConditionalUnetGenerator(nn.Module):
    """U-Net generator with optional condition modulation at selected layers."""

    def __init__(
        self,
        input_nc,
        output_nc,
        num_downs,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        condition_dim=512,
        condition_mode="film",
        condition_layers="bottleneck",
        condition_hidden_dim=128,
        condition_gamma_scale=0.1,
        condition_norm="existing",
        condition_gate_init=0.05,
        condition_residual_alpha_init=0.05,
        condition_num_tokens=8,
        condition_attention_dim=64,
        condition_cross_attention_gate_init=None,
        condition_attention_output_init_std=2e-2,
        condition_attention_logit_scale=4.0,
        condition_attention_direct_tokens=False,
        condition_attention_use_position=False,
    ):
        super(ConditionalUnetGenerator, self).__init__()
        condition_mode = condition_mode.lower()
        if condition_mode == "adain":
            condition_mode = "strict_adain"
        if condition_mode == "strict_adain_bottleneck":
            condition_mode = "strict_adain"
        use_strict_adain_allnorm = condition_mode == "strict_adain_allnorm"
        if use_strict_adain_allnorm:
            condition_layers = "allnorm"
        use_spatial_residual_branch_v2 = condition_mode == "spatial_residual_branch_v2"
        if use_spatial_residual_branch_v2:
            condition_layers = "spatial_residual_v2"
        use_cross_attention_v2 = condition_mode == "cross_attention_v2"
        use_cross_attention_v3 = condition_mode == "cross_attention_v3"
        use_cross_attention_v4 = condition_mode == "cross_attention_v4"
        use_cross_attention_v5 = condition_mode == "cross_attention_v5"
        use_feature_cross_attention = (
            use_cross_attention_v2
            or use_cross_attention_v3
            or use_cross_attention_v4
            or use_cross_attention_v5
        )
        use_cross_attention = condition_mode in (
            "cross_attention",
            "cross_attention_v2",
            "cross_attention_v3",
            "cross_attention_v4",
            "cross_attention_v5",
        )
        if use_cross_attention:
            condition_layers = "cross_attention"
        if use_feature_cross_attention:
            condition_attention_direct_tokens = True
            condition_attention_use_position = True
        if condition_cross_attention_gate_init is None:
            condition_cross_attention_gate_init = condition_gate_init
        if condition_layers not in ("bottleneck", "gated_multistage", "residual_branch", "allnorm", "spatial_residual_v2", "cross_attention"):
            raise NotImplementedError(
                f"Unsupported condition_layers={condition_layers!r}; "
                "use 'bottleneck', 'gated_multistage', 'residual_branch', 'allnorm', 'spatial_residual_v2', or 'cross_attention'."
            )
        if condition_layers == "allnorm" and not use_strict_adain_allnorm:
            raise NotImplementedError("condition_layers='allnorm' requires condition_mode='strict_adain_allnorm'.")
        if condition_layers == "spatial_residual_v2" and not use_spatial_residual_branch_v2:
            raise NotImplementedError(
                "condition_layers='spatial_residual_v2' requires condition_mode='spatial_residual_branch_v2'."
            )
        if condition_layers == "cross_attention" and not use_cross_attention:
            raise NotImplementedError(
                "condition_layers='cross_attention' requires condition_mode='cross_attention'."
            )
        use_gated_multistage = condition_layers == "gated_multistage"
        use_residual_branch = condition_layers == "residual_branch"
        use_strict_adain = condition_mode == "strict_adain"
        condition_module = "strict_adain" if use_strict_adain else "film"
        condition_trunk = (
            SharedConditionTrunk(condition_dim=condition_dim, hidden_dim=condition_hidden_dim)
            if use_strict_adain_allnorm else None
        )
        self.condition_mode = condition_mode
        self.condition_layers = condition_layers

        unet_block = ConditionalUnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
            condition_dim=condition_dim,
            condition_hidden_dim=condition_hidden_dim,
            condition_gamma_scale=condition_gamma_scale,
            condition_norm=condition_norm,
            apply_condition=(not use_residual_branch and not use_strict_adain_allnorm and not use_cross_attention),
            condition_module=condition_module,
            gated_condition=use_gated_multistage,
            condition_gate_init=condition_gate_init,
            allnorm_condition=use_strict_adain_allnorm,
            condition_trunk=condition_trunk,
            cross_attention_condition=False,
            cross_attention_decoder_feature=use_feature_cross_attention,
            cross_attention_direct_tokens=condition_attention_direct_tokens,
            cross_attention_use_position=condition_attention_use_position,
            condition_num_tokens=condition_num_tokens,
            condition_attention_dim=condition_attention_dim,
            condition_cross_attention_gate_init=condition_cross_attention_gate_init,
            condition_attention_output_init_std=condition_attention_output_init_std,
            condition_attention_logit_scale=condition_attention_logit_scale,
            block_name="innermost",
        )
        for i in range(num_downs - 5):
            unet_block = ConditionalUnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                condition_dim=condition_dim,
                condition_hidden_dim=condition_hidden_dim,
                condition_gamma_scale=condition_gamma_scale,
                condition_norm=condition_norm,
                apply_condition=(use_gated_multistage and not use_strict_adain_allnorm),
                condition_module=condition_module,
                gated_condition=use_gated_multistage,
                condition_gate_init=condition_gate_init,
                allnorm_condition=use_strict_adain_allnorm,
                condition_trunk=condition_trunk,
                cross_attention_condition=False,
                cross_attention_decoder_feature=use_feature_cross_attention,
                cross_attention_direct_tokens=condition_attention_direct_tokens,
                cross_attention_use_position=condition_attention_use_position,
                condition_num_tokens=condition_num_tokens,
                condition_attention_dim=condition_attention_dim,
                condition_cross_attention_gate_init=condition_cross_attention_gate_init,
                condition_attention_output_init_std=condition_attention_output_init_std,
                condition_attention_logit_scale=condition_attention_logit_scale,
                block_name=f"middle_{i}",
            )
        unet_block = ConditionalUnetSkipConnectionBlock(
            ngf * 4,
            ngf * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            condition_dim=condition_dim,
            condition_hidden_dim=condition_hidden_dim,
            condition_gamma_scale=condition_gamma_scale,
            condition_norm=condition_norm,
            apply_condition=(use_gated_multistage and not use_strict_adain_allnorm),
            condition_module=condition_module,
            gated_condition=use_gated_multistage,
            condition_gate_init=condition_gate_init,
            allnorm_condition=use_strict_adain_allnorm,
            condition_trunk=condition_trunk,
            cross_attention_condition=use_cross_attention,
            cross_attention_decoder_feature=use_feature_cross_attention,
            cross_attention_direct_tokens=condition_attention_direct_tokens,
            cross_attention_use_position=condition_attention_use_position,
            condition_num_tokens=condition_num_tokens,
            condition_attention_dim=condition_attention_dim,
            condition_cross_attention_gate_init=condition_cross_attention_gate_init,
            condition_attention_output_init_std=condition_attention_output_init_std,
            condition_attention_logit_scale=condition_attention_logit_scale,
            block_name="scale_4x",
        )
        unet_block = ConditionalUnetSkipConnectionBlock(
            ngf * 2,
            ngf * 4,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            condition_dim=condition_dim,
            condition_hidden_dim=condition_hidden_dim,
            condition_gamma_scale=condition_gamma_scale,
            allnorm_condition=use_strict_adain_allnorm,
            condition_trunk=condition_trunk,
            cross_attention_condition=use_cross_attention,
            cross_attention_decoder_feature=use_feature_cross_attention,
            cross_attention_direct_tokens=condition_attention_direct_tokens,
            cross_attention_use_position=condition_attention_use_position,
            condition_num_tokens=condition_num_tokens,
            condition_attention_dim=condition_attention_dim,
            condition_gate_init=condition_gate_init,
            condition_cross_attention_gate_init=condition_cross_attention_gate_init,
            condition_attention_output_init_std=condition_attention_output_init_std,
            condition_attention_logit_scale=condition_attention_logit_scale,
            block_name="scale_2x",
        )
        unet_block = ConditionalUnetSkipConnectionBlock(
            ngf,
            ngf * 2,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            condition_dim=condition_dim,
            condition_hidden_dim=condition_hidden_dim,
            condition_gamma_scale=condition_gamma_scale,
            allnorm_condition=use_strict_adain_allnorm,
            condition_trunk=condition_trunk,
            cross_attention_condition=use_cross_attention,
            cross_attention_decoder_feature=use_feature_cross_attention,
            cross_attention_direct_tokens=condition_attention_direct_tokens,
            cross_attention_use_position=condition_attention_use_position,
            condition_num_tokens=condition_num_tokens,
            condition_attention_dim=condition_attention_dim,
            condition_gate_init=condition_gate_init,
            condition_cross_attention_gate_init=condition_cross_attention_gate_init,
            condition_attention_output_init_std=condition_attention_output_init_std,
            condition_attention_logit_scale=condition_attention_logit_scale,
            block_name="scale_1x",
        )
        self.model = ConditionalUnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
            condition_dim=condition_dim,
            condition_hidden_dim=condition_hidden_dim,
            condition_gamma_scale=condition_gamma_scale,
            condition_norm=condition_norm,
            residual_branch=use_residual_branch,
            spatial_residual_branch_v2=use_spatial_residual_branch_v2,
            condition_residual_alpha_init=condition_residual_alpha_init,
            allnorm_condition=use_strict_adain_allnorm,
            condition_trunk=condition_trunk,
            cross_attention_condition=use_cross_attention_v2,
            cross_attention_pre_output=use_cross_attention_v5,
            cross_attention_decoder_feature=use_cross_attention_v2,
            cross_attention_direct_tokens=condition_attention_direct_tokens,
            cross_attention_use_position=condition_attention_use_position,
            condition_num_tokens=condition_num_tokens,
            condition_attention_dim=condition_attention_dim,
            condition_cross_attention_gate_init=condition_cross_attention_gate_init,
            condition_attention_output_init_std=condition_attention_output_init_std,
            condition_attention_logit_scale=condition_attention_logit_scale,
            block_name="outermost",
        )

    def forward(self, input, condition=None):
        if condition is None:
            raise ValueError("ConditionalUnetGenerator.forward requires condition=[B,512].")
        return self.model(input, condition)

    def reset_condition_to_identity(self):
        for module in self.modules():
            if isinstance(
                module,
                (
                    ConditionalAffine2d,
                    ConditionalInstanceNorm2d,
                    SharedStrictAdaIN2d,
                    ConditionalLayerNorm,
                    GatedConditionalAffine2d,
                    ConditionalResidualBranch,
                    SpatialResidualBranchV2,
                    CrossAttentionCondition2d,
                ),
            ):
                module.reset_to_identity()

    def zero_last_layer(self):
        self.model.up[1].weight.data *= 0
        self.model.up[1].bias.data *= 0

    def get_condition_residual(self):
        return getattr(self.model, "last_condition_residual", None)

    def get_base_output(self):
        return getattr(self.model, "last_base_output", None)

    def get_condition_residual_alpha(self):
        residual_branch = getattr(self.model, "condition_residual_branch", None)
        if residual_branch is None:
            return None
        return torch.sigmoid(residual_branch.alpha_logit).detach()

    def get_adain_site_table(self):
        rows = []
        for name, module in self.named_modules():
            if isinstance(module, SharedStrictAdaIN2d):
                rows.append(
                    {
                        "module": name,
                        "site": module.site_name,
                        "channels": module.num_features,
                    }
                )
        return rows

    def get_condition_stats(self):
        stats = {}
        for name, module in self.named_modules():
            if isinstance(module, SharedStrictAdaIN2d):
                stats[name] = dict(module.last_stats)
        return stats

    def get_spatial_residual_v2_stats(self):
        branch = getattr(self.model, "spatial_residual_branch_v2", None)
        if branch is None:
            return {}
        return dict(branch.last_stats)

    def get_spatial_residual_v2_grad_stats(self):
        branch = getattr(self.model, "spatial_residual_branch_v2", None)
        if branch is None:
            return {}
        return branch.collect_grad_stats()

    def get_cross_attention_site_table(self):
        rows = []
        for name, module in self.named_modules():
            if isinstance(module, CrossAttentionCondition2d):
                rows.append(
                    {
                        "module": name,
                        "site": module.site_name,
                        "channels": module.num_features,
                        "num_tokens": module.num_tokens,
                        "attention_dim": module.attention_dim,
                        "num_heads": module.num_heads,
                        "gate_init": float(torch.sigmoid(module.gate_logit.detach()).cpu()),
                        "output_init_std": float(module.output_init_std),
                        "logit_scale": float(
                            module.logit_scale_log.detach().exp().clamp(0.1, 32.0).cpu()
                        ),
                        "normalize_qk": bool(module.normalize_qk),
                        "direct_token_projection": bool(module.direct_token_projection),
                        "use_spatial_position": bool(module.use_spatial_position),
                        "parameters": sum(param.numel() for param in module.parameters()),
                    }
                )
        return rows

    def get_cross_attention_decoder_tail_table(self):
        """Return the small decoder tail that turns conditioned features into UV offsets."""
        blocks = (
            ("scale_2x_up", self.model.submodule.submodule.up),
            ("scale_1x_up", self.model.submodule.up),
            ("deformation_output", self.model.up),
        )
        rows = []
        for name, module in blocks:
            rows.append(
                {
                    "name": name,
                    "module": module,
                    "parameters": sum(param.numel() for param in module.parameters()),
                }
            )
        return rows

    def get_cross_attention_full_decoder_table(self):
        """Return every recursive decoder up block without including the encoder."""
        rows = []
        seen_modules = set()
        for module in self.modules():
            if not isinstance(module, ConditionalUnetSkipConnectionBlock):
                continue
            up = module.up
            if id(up) in seen_modules:
                continue
            seen_modules.add(id(up))
            rows.append(
                {
                    "name": f"{module.block_name}_up",
                    "module": up,
                    "parameters": sum(param.numel() for param in up.parameters()),
                }
            )
        return rows

    def get_cross_attention_stats(self):
        stats = {}
        for name, module in self.named_modules():
            if isinstance(module, CrossAttentionCondition2d):
                stats[name] = dict(module.last_stats)
        return stats

    def get_cross_attention_regularization(self, entropy_target_ratio=0.72):
        """Return active-condition selectivity and global token-usage penalties."""
        entropy_target_ratio = float(entropy_target_ratio)
        selectivity_terms = []
        balance_terms = []
        for module in self.modules():
            if not isinstance(module, CrossAttentionCondition2d):
                continue
            entropy_ratio = module.last_active_entropy_ratio
            balance_loss = module.last_token_balance_loss
            if entropy_ratio is None or balance_loss is None:
                continue
            selectivity_terms.append(
                F.relu(entropy_ratio - entropy_target_ratio).square()
            )
            balance_terms.append(balance_loss)
        if not selectivity_terms:
            parameter = next(self.parameters())
            zero = parameter.sum() * 0.0
            return zero, zero
        return torch.stack(selectivity_terms).mean(), torch.stack(balance_terms).mean()

    def get_cross_attention_grad_stats(self):
        stats = {}
        for name, module in self.named_modules():
            if not isinstance(module, CrossAttentionCondition2d):
                continue
            prefix = f"condition/cross_attention/{module.site_name.replace('.', '/')}"
            stats[f"{prefix}/token_grad_norm"] = SpatialResidualBranchV2._grad_norm(
                module.condition_to_tokens.parameters()
            )
            stats[f"{prefix}/query_grad_norm"] = SpatialResidualBranchV2._grad_norm(
                module.query_proj.parameters()
            )
            if module.position_proj is not None:
                stats[f"{prefix}/position_grad_norm"] = SpatialResidualBranchV2._grad_norm(
                    module.position_proj.parameters()
                )
            stats[f"{prefix}/key_grad_norm"] = SpatialResidualBranchV2._grad_norm(
                module.key_proj.parameters()
            )
            stats[f"{prefix}/key_token_bias_grad_norm"] = SpatialResidualBranchV2._grad_norm(
                [module.key_token_bias]
            )
            stats[f"{prefix}/value_grad_norm"] = SpatialResidualBranchV2._grad_norm(
                module.value_proj.parameters()
            )
            stats[f"{prefix}/out_grad_norm"] = SpatialResidualBranchV2._grad_norm(
                module.out_proj.parameters()
            )
            stats[f"{prefix}/gate_grad_norm"] = SpatialResidualBranchV2._grad_norm(
                [module.gate_logit]
            )
            stats[f"{prefix}/logit_scale_grad_norm"] = SpatialResidualBranchV2._grad_norm(
                [module.logit_scale_log]
            )
        return stats


class ConditionalUnetSkipConnectionBlock(nn.Module):
    """Conditional U-Net block with explicit recursion for condition passing."""

    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        condition_dim=512,
        condition_hidden_dim=128,
        condition_gamma_scale=0.1,
        condition_norm="existing",
        apply_condition=False,
        condition_module="film",
        gated_condition=False,
        condition_gate_init=0.05,
        residual_branch=False,
        spatial_residual_branch_v2=False,
        condition_residual_alpha_init=0.05,
        allnorm_condition=False,
        condition_trunk=None,
        cross_attention_condition=False,
        cross_attention_pre_output=False,
        cross_attention_decoder_feature=False,
        cross_attention_direct_tokens=False,
        cross_attention_use_position=False,
        condition_num_tokens=8,
        condition_attention_dim=64,
        condition_cross_attention_gate_init=None,
        condition_attention_output_init_std=2e-2,
        condition_attention_logit_scale=4.0,
        block_name=None,
    ):
        super(ConditionalUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.submodule = submodule
        self.dropout = nn.Dropout(0.5) if use_dropout else None
        self.condition_residual_branch = None
        self.spatial_residual_branch_v2 = None
        self.cross_attention_condition = None
        self.pre_output_cross_attention_condition = None
        self.cross_attention_decoder_feature = bool(cross_attention_decoder_feature)
        self.last_condition_residual = None
        self.last_base_output = None
        self.allnorm_condition = allnorm_condition
        self.block_name = block_name or "block"
        self.downnorm_condition = None
        self.upnorm_condition = None
        if condition_cross_attention_gate_init is None:
            condition_cross_attention_gate_init = condition_gate_init

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        self.condition_affine = None

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            self.down = nn.Sequential(downconv)
            self.up = nn.Sequential(uprelu, upconv)
            if residual_branch:
                self.condition_residual_branch = ConditionalResidualBranch(
                    inner_nc * 2,
                    outer_nc,
                    condition_dim=condition_dim,
                    hidden_dim=condition_hidden_dim,
                    gamma_scale=condition_gamma_scale,
                    norm_type=condition_norm,
                    alpha_init=condition_residual_alpha_init,
                )
            if spatial_residual_branch_v2:
                self.spatial_residual_branch_v2 = SpatialResidualBranchV2(
                    input_channels=input_nc,
                    output_channels=outer_nc,
                    scale_1x_channels=inner_nc * 2,
                    scale_2x_channels=inner_nc * 4,
                    condition_dim=condition_dim,
                    hidden_dim=condition_hidden_dim,
                    gamma_scale=condition_gamma_scale,
                )
            if cross_attention_condition:
                self.cross_attention_condition = self._make_cross_attention_module(
                    outer_nc,
                    condition_dim,
                    condition_hidden_dim,
                    condition_attention_dim,
                    condition_num_tokens,
                    condition_cross_attention_gate_init,
                    condition_attention_output_init_std,
                    condition_attention_logit_scale,
                    f"{self.block_name}.output_cross_attention_c{outer_nc}",
                    direct_token_projection=cross_attention_direct_tokens,
                    use_spatial_position=cross_attention_use_position,
                )
            if cross_attention_pre_output:
                self.pre_output_cross_attention_condition = self._make_cross_attention_module(
                    inner_nc * 2,
                    condition_dim,
                    condition_hidden_dim,
                    condition_attention_dim,
                    condition_num_tokens,
                    condition_cross_attention_gate_init,
                    condition_attention_output_init_std,
                    condition_attention_logit_scale,
                    f"{self.block_name}.pre_output_cross_attention_c{inner_nc * 2}",
                    direct_token_projection=cross_attention_direct_tokens,
                    use_spatial_position=cross_attention_use_position,
                )
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            self.down = nn.Sequential(downrelu, downconv)
            if allnorm_condition:
                self.up = nn.Sequential(uprelu, upconv)
                self.upnorm_condition = self._make_allnorm_module(
                    outer_nc,
                    condition_trunk,
                    condition_gamma_scale,
                    f"{self.block_name}.upnorm_c{outer_nc}",
                )
            else:
                self.up = nn.Sequential(uprelu, upconv, upnorm)
            if apply_condition:
                self.condition_affine = self._make_condition_module(
                    outer_nc,
                    condition_dim,
                    condition_hidden_dim,
                    condition_gamma_scale,
                    condition_norm,
                    condition_module,
                    gated_condition,
                    condition_gate_init,
                )
            if cross_attention_condition:
                self.cross_attention_condition = self._make_cross_attention_module(
                    outer_nc if self.cross_attention_decoder_feature else input_nc + outer_nc,
                    condition_dim,
                    condition_hidden_dim,
                    condition_attention_dim,
                    condition_num_tokens,
                    condition_cross_attention_gate_init,
                    condition_attention_output_init_std,
                    condition_attention_logit_scale,
                    (
                        f"{self.block_name}.decoder_cross_attention_c{outer_nc}"
                        if self.cross_attention_decoder_feature
                        else f"{self.block_name}.skip_cross_attention_c{input_nc + outer_nc}"
                    ),
                    direct_token_projection=cross_attention_direct_tokens,
                    use_spatial_position=cross_attention_use_position,
                )
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            if allnorm_condition:
                self.down = nn.Sequential(downrelu, downconv)
                self.up = nn.Sequential(uprelu, upconv)
                self.downnorm_condition = self._make_allnorm_module(
                    inner_nc,
                    condition_trunk,
                    condition_gamma_scale,
                    f"{self.block_name}.downnorm_c{inner_nc}",
                )
                self.upnorm_condition = self._make_allnorm_module(
                    outer_nc,
                    condition_trunk,
                    condition_gamma_scale,
                    f"{self.block_name}.upnorm_c{outer_nc}",
                )
            else:
                self.down = nn.Sequential(downrelu, downconv, downnorm)
                self.up = nn.Sequential(uprelu, upconv, upnorm)
            if apply_condition:
                self.condition_affine = self._make_condition_module(
                    outer_nc,
                    condition_dim,
                    condition_hidden_dim,
                    condition_gamma_scale,
                    condition_norm,
                    condition_module,
                    gated_condition,
                    condition_gate_init,
                )
            if cross_attention_condition:
                self.cross_attention_condition = self._make_cross_attention_module(
                    outer_nc if self.cross_attention_decoder_feature else input_nc + outer_nc,
                    condition_dim,
                    condition_hidden_dim,
                    condition_attention_dim,
                    condition_num_tokens,
                    condition_cross_attention_gate_init,
                    condition_attention_output_init_std,
                    condition_attention_logit_scale,
                    (
                        f"{self.block_name}.decoder_cross_attention_c{outer_nc}"
                        if self.cross_attention_decoder_feature
                        else f"{self.block_name}.skip_cross_attention_c{input_nc + outer_nc}"
                    ),
                    direct_token_projection=cross_attention_direct_tokens,
                    use_spatial_position=cross_attention_use_position,
                )

    @staticmethod
    def _make_allnorm_module(
        num_features,
        condition_trunk,
        condition_gamma_scale,
        site_name,
    ):
        if condition_trunk is None:
            raise ValueError("strict_adain_allnorm requires a shared condition trunk.")
        return SharedStrictAdaIN2d(
            num_features,
            condition_trunk=condition_trunk,
            gamma_scale=condition_gamma_scale,
            site_name=site_name,
        )

    @staticmethod
    def _make_condition_module(
        outer_nc,
        condition_dim,
        condition_hidden_dim,
        condition_gamma_scale,
        condition_norm,
        condition_module,
        gated_condition,
        condition_gate_init,
    ):
        if gated_condition:
            return GatedConditionalAffine2d(
                outer_nc,
                condition_dim=condition_dim,
                hidden_dim=condition_hidden_dim,
                gamma_scale=condition_gamma_scale,
                norm_type=condition_norm,
                gate_init=condition_gate_init,
            )
        if condition_module == "strict_adain":
            return ConditionalInstanceNorm2d(
                outer_nc,
                condition_dim=condition_dim,
                hidden_dim=condition_hidden_dim,
                gamma_scale=condition_gamma_scale,
            )
        if condition_module == "film":
            return ConditionalAffine2d(
                outer_nc,
                condition_dim=condition_dim,
                hidden_dim=condition_hidden_dim,
                gamma_scale=condition_gamma_scale,
                norm_type=condition_norm,
            )
        raise NotImplementedError(f"Unsupported condition_module={condition_module!r}")

    @staticmethod
    def _make_cross_attention_module(
        num_features,
        condition_dim,
        condition_hidden_dim,
        condition_attention_dim,
        condition_num_tokens,
        condition_gate_init,
        condition_attention_output_init_std,
        condition_attention_logit_scale,
        site_name,
        direct_token_projection=False,
        use_spatial_position=False,
    ):
        return CrossAttentionCondition2d(
            num_features,
            condition_dim=condition_dim,
            hidden_dim=condition_hidden_dim,
            num_tokens=condition_num_tokens,
            attention_dim=condition_attention_dim,
            residual_gate_init=condition_gate_init,
            output_init_std=condition_attention_output_init_std,
            logit_scale_init=condition_attention_logit_scale,
            site_name=site_name,
            direct_token_projection=direct_token_projection,
            use_spatial_position=use_spatial_position,
        )

    def _forward_down(self, x, condition):
        down = self.down(x)
        if self.downnorm_condition is not None:
            down = self.downnorm_condition(down, condition)
        return down

    def _forward_up(self, x, condition):
        up = self.up(x)
        if self.upnorm_condition is not None:
            up = self.upnorm_condition(up, condition)
        return up

    def forward(self, x, condition, feature_collector=None):
        if self.outermost and self.spatial_residual_branch_v2 is not None:
            feature_collector = {}
        down = self._forward_down(x, condition)
        if self.innermost:
            up = self._forward_up(down, condition)
            if self.condition_affine is not None:
                up = self.condition_affine(up, condition)
            if self.cross_attention_condition is not None and self.cross_attention_decoder_feature:
                up = self.cross_attention_condition(up, condition)
            out = torch.cat([x, up], 1)
            if self.cross_attention_condition is not None and not self.cross_attention_decoder_feature:
                out = self.cross_attention_condition(out, condition)
            if feature_collector is not None:
                feature_collector[self.block_name] = out
            return out

        if self.submodule is not None:
            down = self.submodule(down, condition, feature_collector=feature_collector)
        if self.outermost and self.pre_output_cross_attention_condition is not None:
            down = self.pre_output_cross_attention_condition(down, condition)
        up = self._forward_up(down, condition)
        if self.dropout is not None:
            up = self.dropout(up)

        if self.outermost:
            self.last_condition_residual = None
            self.last_base_output = up
            if self.cross_attention_condition is not None:
                conditioned_up = self.cross_attention_condition(up, condition)
                self.last_condition_residual = conditioned_up - up
                return conditioned_up
            if self.condition_residual_branch is not None:
                condition_residual = self.condition_residual_branch(down, condition)
                if condition_residual.shape != up.shape:
                    raise ValueError(
                        "Condition residual shape must match baseline deformation output: "
                        f"residual={tuple(condition_residual.shape)}, base={tuple(up.shape)}"
                    )
                self.last_condition_residual = condition_residual
                return up + condition_residual
            if self.spatial_residual_branch_v2 is not None:
                condition_residual = self.spatial_residual_branch_v2(feature_collector, x, condition)
                if condition_residual.shape != up.shape:
                    raise ValueError(
                        "Spatial residual v2 shape must match baseline deformation output: "
                        f"residual={tuple(condition_residual.shape)}, base={tuple(up.shape)}"
                    )
                self.last_condition_residual = condition_residual
                return up + condition_residual
            return up
        if self.cross_attention_condition is not None and self.cross_attention_decoder_feature:
            up = self.cross_attention_condition(up, condition)
        out = torch.cat([x, up], 1)
        if self.cross_attention_condition is not None and not self.cross_attention_decoder_feature:
            out = self.cross_attention_condition(out, condition)
        if feature_collector is not None:
            feature_collector[self.block_name] = out
        return out


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            # up = [uprelu, upconv, nn.Tanh()]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
