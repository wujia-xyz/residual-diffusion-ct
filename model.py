import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from denoising_diffusion_pytorch.attend import Attend


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)


def divisible_by(numer, denom):
    return (numer % denom) == 0


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class Block(Module):
    def __init__(self, dim, dim_out, dropout=0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)


class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4, flash=False):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash=flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class HFProcessor(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.channel_mapper = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            RMSNorm(out_channels // 2),
            nn.SiLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.channel_mapper(x)


class Unet(Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        input_channels=1,
        hf_channels=1,
        self_condition=False,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        dropout=0.,
        attn_dim_head=32,
        attn_heads=4,
        full_attn=None,
        flash_attn=False
    ):
        super().__init__()

        self.channels = channels
        self.hf_channels = hf_channels
        self.self_condition = self_condition
        total_input_channels = input_channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(total_input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash=flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout)

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
                zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        self.ups = ModuleList([])
        for ind in reversed(range(num_resolutions)):
            layer_full_attn = full_attn[ind]
            layer_attn_heads = attn_heads[ind]
            layer_attn_dim_head = attn_dim_head[ind]

            dim_in, dim_out = in_out[ind]
            skip_channels = dim_in

            upsample_in_ch = mid_dim if ind == (num_resolutions - 1) else in_out[ind + 1][0]

            upsample_module = Upsample(upsample_in_ch, skip_channels)

            block1 = resnet_block(skip_channels * 3, skip_channels)
            block2 = resnet_block(skip_channels * 2, skip_channels)

            attn_klass = FullAttention if layer_full_attn else LinearAttention
            attn_module = attn_klass(skip_channels, dim_head=layer_attn_dim_head, heads=layer_attn_heads)

            self.ups.append(ModuleList([
                upsample_module,
                block1,
                block2,
                attn_module
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        final_block_input_dim = init_dim * 3
        self.final_res_block = resnet_block(final_block_input_dim, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

        self.hf_init_conv = nn.Conv2d(hf_channels, init_dim, 7, padding=3)
        self.hf_processors = ModuleList([])

        for dim_in, _ in in_out:
            self.hf_processors.append(HFProcessor(dim_in, dim_in))

        self.final_hf_processor = HFProcessor(init_dim, init_dim)

    @property
    def downsample_factor(self):
        factor = 1
        for module_list in self.downs[:-1]:
            last_op = module_list[-1]
            is_downsample = isinstance(last_op, nn.Sequential) and \
                            len(last_op) > 0 and \
                            isinstance(last_op[0], Rearrange) and \
                            'p1 = 2' in last_op[0].pattern and \
                            'p2 = 2' in last_op[0].pattern
            if is_downsample:
                factor *= 2
        return factor

    def forward(self, x, x_hf, time, x_self_cond=None):
        if not all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]):
            print(f"Warning: Input dimensions {x.shape[-2:]} might not be perfectly divisible by {self.downsample_factor}.")
        assert x.shape[0] == x_hf.shape[0], "Batch size mismatch"
        assert x.shape[-2:] == x_hf.shape[-2:], "Spatial dimension mismatch"

        x_orig = x

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x_orig))
            x = torch.cat((x_self_cond, x_orig), dim=1)
        else:
            x = x_orig

        t = self.time_mlp(time)

        x = self.init_conv(x)
        r = x.clone()

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x) + x
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        hf = self.hf_init_conv(x_hf)

        hf_features = [hf]
        hf_current = hf

        for _, _, _, downsample in self.downs:
            hf_current = downsample(hf_current)
            hf_features.append(hf_current)

        for i, (upsample, block1, block2, attn) in enumerate(self.ups):
            h_idx = len(h) - i - 1
            enc_features = h[h_idx]

            x = upsample(x)

            if x.shape[-2:] != enc_features.shape[-2:]:
                x = F.interpolate(x, size=enc_features.shape[-2:], mode='bilinear', align_corners=False)

            hf_idx = h_idx
            hf_feature = hf_features[hf_idx]
            hf_processed = self.hf_processors[h_idx](hf_feature)

            if hf_processed.shape[-2:] != enc_features.shape[-2:]:
                hf_processed = F.interpolate(hf_processed, size=enc_features.shape[-2:], mode='bilinear', align_corners=False)

            x_cat = torch.cat((x, enc_features, hf_processed), dim=1)

            x = block1(x_cat, t)

            x_cat2 = torch.cat((x, enc_features), dim=1)
            x = block2(x_cat2, t)

            x = attn(x) + x

        final_hf = self.final_hf_processor(hf_features[0])

        x = torch.cat((x, r, final_hf), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, first_block=False):
        super(DownBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(nn.Conv2d(1, in_ch, 3, 1, 1), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True)) if first_block else nn.MaxPool2d(2),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.model(x)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.pool = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, 2, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        x_t = self.pool(x)
        x_in = torch.cat((x_t, y), dim=1)
        out = self.model(x_in)
        return out


class FBPConvNet(nn.Module):
    def __init__(self):
        super(FBPConvNet, self).__init__()
        self.conv1 = DownBlock(64, 64, True)
        self.conv2 = DownBlock(64, 128)
        self.conv3 = DownBlock(128, 256)
        self.conv4 = DownBlock(256, 512)
        self.conv5 = DownBlock(512, 1024)
        self.conv4_t = UpBlock(1024, 512)
        self.conv3_t = UpBlock(512, 256)
        self.conv2_t = UpBlock(256, 128)
        self.conv1_t = UpBlock(128, 64)
        self.conv_last = nn.Conv2d(64, 1, 1, 1, 0)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        x_3 = self.conv3(x_2)
        x_4 = self.conv4(x_3)
        x_5 = self.conv5(x_4)

        y_4 = self.conv4_t(x_5, x_4)
        y_3 = self.conv3_t(y_4, x_3)
        y_2 = self.conv2_t(y_3, x_2)
        y_1 = self.conv1_t(y_2, x_1)
        y = self.conv_last(y_1)
        out = y
        return out
