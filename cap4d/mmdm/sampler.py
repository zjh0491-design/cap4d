from typing import Tuple, Dict
import torch
import numpy as np
from tqdm import tqdm
from controlnet.ldm.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
)


class StochasticIOSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()

        if isinstance(model, dict):
            # model distributed on different devices
            self.device_model_map = model
        else:
            if torch.cuda.is_available():
                self.device_model_map = {"cuda": model}
            else:
                self.device_model_map = {"cpu": model}

        for key in self.device_model_map:
            self.main_model = self.device_model_map[key]
            self.ddpm_num_timesteps = self.main_model.num_timesteps
            break

    def register_buffer(self, name, attr):
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.main_model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, "alphas have to be defined for each timestep"
        to_torch = lambda x: x.clone().detach().to(torch.float32)

        self.register_buffer("betas", to_torch(self.main_model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(self.main_model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.detach().cpu())))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod.detach().cpu())))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod.detach().cpu())))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.detach().cpu())))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.detach().cpu() - 1.0)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.detach().cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer("ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(
        self,
        S: int,
        ref_cond: Dict[str, torch.Tensor],
        ref_uncond: Dict[str, torch.Tensor],
        gen_cond: Dict[str, torch.Tensor],
        gen_uncond: Dict[str, torch.Tensor],
        latent_shape: Tuple[int, int, int],
        V: int = 8,
        R_max: int = 4,
        cfg_scale: float = 1.0,
        eta: float = 0.0,
        verbose: bool = False,
    ):
        mem_device = next(iter(gen_cond.items()))[1].device
        n_devices = len(self.device_model_map)

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)

        n_gen = next(iter(gen_cond.items()))[1].shape[0]
        n_all_ref = next(iter(ref_cond.items()))[1].shape[0]
        R = min(n_all_ref, R_max)

        assert n_gen % (V - R) == 0, f"number of generated images ({n_gen}) has to be divisible by G ({V-R})"
        n_its = n_gen // (V - R)

        all_x_T = torch.randn((n_gen, *latent_shape), device=mem_device)
        all_e_t = torch.zeros_like(all_x_T)

        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        print(f"Running stochastic I/O sampling with {total_steps} timesteps, {R} reference images and {n_gen} generated images")

        iterator = tqdm(time_range, desc="Stochastic I/O sampler", total=total_steps)

        def dict_sample(in_dict, indices, device=None, add_batch_dim=True):
            out_dict = {}
            idx = torch.as_tensor(indices, dtype=torch.long)
            for key, value in in_dict.items():
                local_idx = idx.to(value.device)
                out = torch.index_select(value, 0, local_idx)
                if add_batch_dim:
                    out = out.unsqueeze(0)
                if device is not None:
                    out = out.to(device)
                out_dict[key] = out
            return out_dict

        # --------------------------
        # 1️⃣ 按 GPU 分配 batch
        # --------------------------
        batch_indices = []
        for l in range(int(np.ceil(n_its / n_devices))):
            device_batch = []
            for dev_id in range(n_devices):
                it_idx = l * n_devices + dev_id
                if it_idx < n_its:
                    device_batch.append(it_idx)
            if len(device_batch) > 0:
                batch_indices.append(device_batch)

        # --------------------------
        # 2️⃣ GPU 并行采样
        # --------------------------
        streams = {dev_key: torch.cuda.Stream(device=self.device_model_map[dev_key].device)
                   for dev_key in self.device_model_map}

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((1, V), step, device=mem_device, dtype=torch.long)
            all_e_t = all_e_t * 0.0

            if R == 1:
                ref_batches = np.zeros((n_its, R), dtype=np.int64)
            else:
                ref_batches = np.stack(
                    [np.random.permutation(np.arange(n_all_ref))[:R] for _ in range(n_its)],
                    axis=0,
                )

            gen_batches = np.reshape(np.random.permutation(np.arange(n_gen)), (n_its, -1))

            results_per_device = {}

            for dev_batches in batch_indices:
                # 每张卡并行执行
                for dev_id, it_idx in enumerate(dev_batches):
                    dev_key = list(self.device_model_map)[dev_id]
                    dev_model = self.device_model_map[dev_key]
                    dev_device = dev_model.device

                    with torch.cuda.stream(streams[dev_key]):
                        ref_indices = ref_batches[it_idx]
                        gen_indices = gen_batches[it_idx]

                        curr_ref_cond = dict_sample(ref_cond, ref_indices, device=dev_device)
                        curr_gen_cond = dict_sample(gen_cond, gen_indices, device=dev_device)
                        curr_ref_uncond = dict_sample(ref_uncond, ref_indices, device=dev_device)
                        curr_gen_uncond = dict_sample(gen_uncond, gen_indices, device=dev_device)

                        curr_cond = {}
                        curr_uncond = {}
                        c_in = {}

                        for key in curr_ref_cond:
                            curr_cond[key] = torch.cat([curr_ref_cond[key], curr_gen_cond[key]], dim=1)
                            curr_uncond[key] = torch.cat([curr_ref_uncond[key], curr_gen_uncond[key]], dim=1)
                            c_in[key] = torch.cat([curr_uncond[key], curr_cond[key]], dim=0)

                        t_in = torch.cat([ts]*2, dim=0).to(dev_device)
                        c_in = dict(c_concat=[c_in])

                        x_in = torch.cat([curr_cond["z_input"][:, :R], 
                                          torch.index_select(all_x_T, 0, torch.as_tensor(gen_indices, dtype=torch.long, device=all_x_T.device)).unsqueeze(0).to(dev_device)], dim=1)
                        x_in = torch.cat([x_in]*2, dim=0).to(dev_device)

                        model_uncond, model_t = dev_model.apply_model(x_in, t_in, c_in).chunk(2)
                        model_output = model_uncond + cfg_scale * (model_t - model_uncond)

                        e_t = model_output[:, R:].squeeze(0)  # 提取生成部分并移除 batch 维
                        results_per_device[dev_key] = (gen_indices, e_t)

                # 等待所有 stream 完成
                for dev_key in streams:
                    torch.cuda.synchronize(dev_key)

                # 收集结果
                for dev_key, (gen_indices, e_t) in results_per_device.items():
                    all_e_t.index_add_(0, torch.as_tensor(gen_indices, dtype=torch.long, device=all_e_t.device),
                                       e_t.to(mem_device))

            alpha_t = self.ddim_alphas.float()[index]
            sqrt_one_minus_alpha_t = self.ddim_sqrt_one_minus_alphas[index]
            sigma_t = self.ddim_sigmas[index]
            alpha_prev_t = torch.tensor(self.ddim_alphas_prev).float()[index]

            alpha_prev_t = alpha_prev_t.double()
            sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.double()
            alpha_t = alpha_t.double()
            alpha_prev_t = alpha_prev_t.double()

            e_t_factor = (
                -alpha_prev_t.sqrt() * sqrt_one_minus_alpha_t / alpha_t.sqrt()
                + (1.0 - alpha_prev_t - sigma_t**2).sqrt()
            )
            x_t_factor = alpha_prev_t.sqrt() / alpha_t.sqrt()

            e_t_factor = e_t_factor.float()
            x_t_factor = x_t_factor.float()

            all_x_T = all_x_T * x_t_factor + all_e_t * e_t_factor

        return all_x_T