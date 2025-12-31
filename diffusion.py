import enum
import math

import torch
import numpy as np
import torch as th
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn

from basic_ops import mean_flat
from losses import normal_kl, discretized_gaussian_log_likelihood
from torch_radon import RadonFanbeam
from edgeloss import Eagle_Loss
from pytorch_wavelets import DWTForward, DWTInverse


eagle_loss = Eagle_Loss(patch_size=3)
dwt = DWTForward(J=1, mode='zero', wave='db1').cuda()
idwt = DWTInverse(mode='zero', wave='db1').cuda()

vgg16 = models.vgg16(pretrained=True).features.cuda().eval()
for param in vgg16.parameters():
    param.requires_grad = False

angles = torch.linspace(0, 2 * torch.pi, 64)
resolution = 512
radon = RadonFanbeam(
    resolution,
    angles,
    source_distance=590.0,
    det_distance=490.0,
    det_count=736,
    det_spacing=2.1,
    clip_to_circle=False
)


def extract_high_frequency(image):
    low_freq, high_freq = dwt(image)
    low_freq_zeros = torch.zeros_like(low_freq)
    high_freq_image = idwt((low_freq_zeros, high_freq))
    return high_freq_image


def extract_low_frequency(image):
    low_freq, high_freq = dwt(image)
    high_freq_zeros = []
    for h in high_freq:
        high_freq_zeros.append(torch.zeros_like(h))
    low_freq_image = idwt((low_freq, high_freq_zeros))
    return low_freq_image


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_weights=None):
        super(VGGPerceptualLoss, self).__init__()

        if layer_weights is None:
            self.layer_weights = {
                '3': 1.0,
                '8': 1.0,
                '15': 1.0,
            }
        else:
            self.layer_weights = layer_weights

        self.criterion = nn.L1Loss()

    def create_three_channel_input(self, x):
        original = x
        high_freq = extract_high_frequency(x)
        low_freq = extract_low_frequency(x)

        if original.shape[1] > 1:
            original = original[:, 0:1, :, :]
        if high_freq.shape[1] > 1:
            high_freq = high_freq[:, 0:1, :, :]
        if low_freq.shape[1] > 1:
            low_freq = low_freq[:, 0:1, :, :]

        three_channel = torch.cat([original, high_freq, low_freq], dim=1)
        return three_channel

    def forward(self, output, target):
        output_3ch = self.create_three_channel_input(output)
        target_3ch = self.create_three_channel_input(target)

        output_3ch = (output_3ch + 1) / 2.0
        target_3ch = (target_3ch + 1) / 2.0

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        output_3ch = (output_3ch - mean) / std
        target_3ch = (target_3ch - mean) / std

        loss = 0.0

        x = output_3ch
        y = target_3ch

        for name, layer in enumerate(vgg16):
            x = layer(x)
            y = layer(y)

            name = str(name)
            if name in self.layer_weights:
                loss += self.layer_weights[name] * self.criterion(x, y)

        return loss


def charbonnier_loss(tensor, epsilon=1e-6):
    return torch.mean(torch.sqrt(tensor ** 2 + epsilon ** 2))


def get_named_eta_schedule(
        schedule_name,
        num_diffusion_timesteps,
        min_noise_level,
        etas_end=0.99,
        kappa=1.0,
        kwargs=None):
    if schedule_name == 'exponential':
        power = kwargs.get('power', None)
        etas_start = min(min_noise_level / kappa, min_noise_level)
        increaser = math.exp(1/(num_diffusion_timesteps-1)*math.log(etas_end/etas_start))
        base = np.ones([num_diffusion_timesteps, ]) * increaser
        power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True)**power
        power_timestep *= (num_diffusion_timesteps-1)
        sqrt_etas = np.power(base, power_timestep) * etas_start
    elif schedule_name == 'ldm':
        import scipy.io as sio
        mat_path = kwargs.get('mat_path', None)
        sqrt_etas = sio.loadmat(mat_path)['sqrt_etas'].reshape(-1)
    else:
        raise ValueError(f"Unknown schedule_name {schedule_name}")

    return sqrt_etas


class ModelMeanType(enum.Enum):
    START_X = enum.auto()
    EPSILON = enum.auto()
    PREVIOUS_X = enum.auto()
    RESIDUAL = enum.auto()
    EPSILON_SCALE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()
    WEIGHTED_MSE = enum.auto()


class ModelVarTypeDDPM(enum.Enum):
    LEARNED = enum.auto()
    LEARNED_RANGE = enum.auto()
    FIXED_LARGE = enum.auto()
    FIXED_SMALL = enum.auto()


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class GaussianDiffusion:
    def __init__(
        self,
        *,
        sqrt_etas,
        kappa,
        model_mean_type=ModelMeanType.START_X,
        loss_type=LossType.MSE,
        sf=4,
        scale_factor=None,
        normalize_input=True,
        latent_flag=True,
    ):
        self.kappa = kappa
        self.model_mean_type = model_mean_type
        self.loss_type = loss_type
        self.scale_factor = scale_factor
        self.normalize_input = normalize_input
        self.latent_flag = latent_flag
        self.sf = sf

        self.perceptual_loss = VGGPerceptualLoss()

        self.sqrt_etas = sqrt_etas
        self.etas = sqrt_etas**2
        assert len(self.etas.shape) == 1, "etas must be 1-D"
        assert (self.etas > 0).all() and (self.etas <= 1).all()

        self.num_timesteps = int(self.etas.shape[0])
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev

        self.posterior_variance = kappa**2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(
            self.posterior_variance[1], self.posterior_variance[1:]
        )
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas

        weight_loss_mse = 0.5 / self.posterior_variance_clipped * (self.alpha / self.etas)**2
        self.weight_loss_mse = weight_loss_mse

    def q_mean_variance(self, x_start, y, t):
        mean = _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
        variance = _extract_into_tensor(self.etas, t, x_start.shape) * self.kappa**2
        log_variance = variance.log()
        return mean, variance, log_variance

    def q_sample(self, x_start, y, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
            + _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_start
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x_t, y, t, model_high=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        y_high = extract_high_frequency(y)

        if model_high is not None:
            with torch.no_grad():
                y_high = model_high(y_high)

        model_input = x_t
        model_output = model(self._scale_input(model_input, t), y_high, t, **model_kwargs)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(model_output)

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, t=t
        )

        assert (
            model_mean.shape == model_output.shape == pred_xstart.shape == x_t.shape
        )
        return {
            "mean": model_mean,
            "variance": _extract_into_tensor(self.posterior_variance, t, x_t.shape),
            "log_variance": _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape),
            "pred_xstart": pred_xstart,
        }

    def p_sample(self, model, x, y, t, model_high=None, clip_denoised=True, denoised_fn=None, model_kwargs=None, noise_repeat=False):
        out = self.p_mean_variance(
            model,
            x,
            y,
            t,
            model_high=model_high,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        if noise_repeat:
            noise = noise[0,].repeat(x.shape[0], 1, 1, 1)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean": out["mean"]}

    def p_sample_loop(
        self,
        x_0,
        y,
        model,
        first_stage_model=None,
        consistencydecoder=None,
        model_high=None,
        noise=None,
        noise_repeat=False,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        regularize=True,
    ):
        final = None
        for sample in self.p_sample_loop_progressive(
            x_0,
            y,
            model,
            first_stage_model=first_stage_model,
            model_high=model_high,
            noise=noise,
            noise_repeat=noise_repeat,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            regularize=regularize,
        ):
            final = sample["sample"]
        with th.no_grad():
            out = self.decode_first_stage(
                final,
                first_stage_model=first_stage_model,
                consistencydecoder=consistencydecoder,
            )
        return out

    def p_sample_loop_progressive(
            self, x_0, y, model,
            first_stage_model=None,
            model_high=None,
            noise=None,
            noise_repeat=False,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            regularize=True,
    ):
        if device is None:
            device = next(model.parameters()).device
        x_0 = x_0.to(device)
        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)

        if noise is None:
            noise = th.randn_like(z_y)
        if noise_repeat:
            noise = noise[0,].repeat(z_y.shape[0], 1, 1, 1)
        z_sample = self.prior_sample(z_y, noise)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * y.shape[0], device=device)

            with th.no_grad():
                out = self.p_sample(
                    model,
                    z_sample,
                    z_y,
                    t,
                    model_high=model_high,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    noise_repeat=noise_repeat,
                )
                z_sample = out["sample"]
                pred_xstart = out["pred_xstart"]

            if regularize and i != 0:
                residual = radon.forward(pred_xstart - x_0)
                d = radon.backward(radon.filter_sinogram(residual))

                Ad = radon.forward(d)
                numerator = (residual * Ad).sum()
                denominator = (Ad * Ad).sum()
                kappa = numerator / (denominator + 1e-8)
                kappa = th.clamp(kappa, 0.0, 1.0)

                print(f"step {i}, kappa = {kappa.item():.4f}")

                z_sample = z_sample - kappa * d

            yield out

    def decode_first_stage(self, z_sample, first_stage_model=None, consistencydecoder=None):
        if first_stage_model is None:
            return z_sample
        batch_size = z_sample.shape[0]
        data_dtype = z_sample.dtype

        if consistencydecoder is None:
            model = first_stage_model
            decoder = first_stage_model.decode
            model_dtype = next(model.parameters()).dtype
        else:
            model = consistencydecoder
            decoder = consistencydecoder
            model_dtype = next(model.ckpt.parameters()).dtype

        z_sample = 1 / self.scale_factor * z_sample
        if consistencydecoder is None:
            out = decoder(z_sample.type(model_dtype))
        else:
            with th.cuda.amp.autocast():
                out = decoder(z_sample)
        if not model_dtype == data_dtype:
            out = out.type(data_dtype)
        return out

    def encode_first_stage(self, y, first_stage_model, up_sample=False):
        if first_stage_model is None:
            return y
        data_dtype = y.dtype
        model_dtype = next(first_stage_model.parameters()).dtype
        if up_sample and self.sf != 1:
            y = F.interpolate(y, scale_factor=self.sf, mode='bicubic')
        if not model_dtype == data_dtype:
            y = y.type(model_dtype)
        with th.no_grad():
            z_y = first_stage_model.encode(y)
            out = z_y * self.scale_factor
        if not model_dtype == data_dtype:
            out = out.type(data_dtype)
        return out

    def prior_sample(self, y, noise=None):
        if noise is None:
            noise = th.randn_like(y)

        t = th.tensor([self.num_timesteps-1,] * y.shape[0], device=y.device).long()

        return y + _extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise

    def training_losses(self, model, x_start, y, t, first_stage_model=None, model_high=None, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}

        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)
        z_start = self.encode_first_stage(x_start, first_stage_model, up_sample=False)

        if noise is None:
            noise = th.randn_like(z_start)

        z_t = self.q_sample(z_start, z_y, t, noise=noise)

        z_y_high = extract_high_frequency(z_y)

        if model_high is not None:
            with torch.no_grad():
                z_y_high = model_high(z_y_high)

        model_input = z_t
        model_output = model(self._scale_input(model_input, t), z_y_high, t, **model_kwargs)
        target = z_start

        assert model_output.shape == target.shape == z_start.shape

        charbonnier = charbonnier_loss(target - model_output)
        eagle = eagle_loss(model_output, target)
        vgg_perceptual = self.perceptual_loss(model_output, target)

        loss = charbonnier + 0.001 * eagle + 0.1 * vgg_perceptual

        return {"loss": loss}

    def _scale_input(self, inputs, t):
        if self.normalize_input:
            if self.latent_flag:
                std = th.sqrt(_extract_into_tensor(self.etas, t, inputs.shape) * self.kappa**2 + 1)
                inputs_norm = inputs / std
            else:
                inputs_max = _extract_into_tensor(self.sqrt_etas, t, inputs.shape) * self.kappa * 3 + 1
                inputs_norm = inputs / inputs_max
        else:
            inputs_norm = inputs
        return inputs_norm
