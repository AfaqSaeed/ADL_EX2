import math
from functools import partial
from einops import rearrange, reduce, repeat
import torch
from torch import nn, einsum
import torch.nn.functional as F
from ex02_helpers import *


# Note: This code employs large parts of the following sources:
# Niels Rogge (nielsr) & Kashif Rasul (kashif): https://huggingface.co/blog/annotated-diffusion (last access: 23.05.2023),
# which is based on
# Phil Wang (lucidrains): https://github.com/lucidrains/denoising-diffusion-pytorch (last access: 23.05.2023)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    # Note: This implements FiLM conditioning, see https://distill.pub/2018/feature-wise-transformations/ and
    # http://arxiv.org/pdf/1709.07871.pdf
    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, classes_emb_dim=None, groups=8):
        super().__init__()
        full_emb_dim = int(default(time_emb_dim, 0)) + int(default(classes_emb_dim, 0))
        # print(f"full_emb_dim {full_emb_dim}, dim_out:{dim_out}")
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(full_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, class_emb=None):

        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            # print(f"time_emb {time_emb.shape},class emb {class_emb.shape}")
            # print(f"time_emb device: {time_emb.device}, class_emb device: {class_emb.device}")
            # print(f"MLP weights device: {next(self.mlp.parameters()).device}")
            cond_emb = torch.cat(cond_emb, dim=-1)
            # print(f"Shape of concatenated embeddings: {cond_emb.shape}")  # Debug added
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1 1')
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


# Linear attention variant, scales linear with sequence length
# Shen et al.: https://arxiv.org/abs/1812.01243
# https://github.com/lucidrains/linear-attention-transformer
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


# Wu et al.: https://arxiv.org/abs/1803.08494
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# TODO: make yourself familiar with the code that is presented here, as it closely interacts with the rest of the exercise.
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=4,
        class_free_guidance=False,  # TODO: Incorporate in your code
        p_uncond=0.2,
        num_classes=10
    ):
        super().__init__()
        self.class_free_guidance = class_free_guidance
        # determine dimensions
        self.channels = channels
        input_channels = channels   # adapted from the original source

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)  # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # TODO: Implement a class embedder for the conditional part of the classifier-free guidance & define a default
        classes_dim = None
        if self.class_free_guidance:
            self.num_classes = num_classes
            self.p_uncond = p_uncond

            self.classes_emb = nn.Embedding(num_classes, dim)
            self.null_classes_emb = nn.Parameter(torch.randn(dim))

            classes_dim = dim * 4

            self.classes_mlp = nn.Sequential(
                nn.Linear(dim, classes_dim),
                nn.GELU(),
                nn.Linear(classes_dim, classes_dim)
            )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # TODO: Adapt all blocks accordingly such that they can accommodate a class embedding as well
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            # print(f"dim in:{dim_in}, dim out{dim_out}")
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim,classes_emb_dim=classes_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim,classes_emb_dim=classes_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim,classes_emb_dim=classes_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, classes=None, p_uncond=None, **kwargs):

        batch, device = x.shape[0], x.device
        c = None

        if self.class_free_guidance and classes is not None:
            classes_emb = self.classes_emb(classes)
            p_uncond = default(p_uncond, self.p_uncond)
            if self.training and p_uncond > 0:

                keep_mask = prob_mask_like((batch,), 1 - p_uncond, device = device)
                null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)
                classes_emb = torch.where(
                    rearrange(keep_mask, 'b -> b 1'),
                    classes_emb,
                    null_classes_emb
                )
        
        c = self.classes_mlp(classes_emb) if classes_emb is not None else None
        # print(f"classese mlp:{c}")

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        # TODO: Implement the class conditioning. Keep in mind that
        #  - for each element in the batch, the class embedding is replaced with the null token with a certain probability during training
        #  - during testing, you need to have control over whether the conditioning is applied or not
        # #  - analogously to the time embedding, the class embedding is provided in every ResNet block as additional conditioning
        # if classes is not None and self.training:
        #     if torch.rand(1).item() < self.p_uncond:
        #         labels = torch.tensor([self.num_classes] * classes.size(0), device=classes.device)  # Use null token
        # c = self.classes_emb(labels) if classes is not None else 0


        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, c)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)

    def forward_with_guidance(
        self,
        *args,
        guidance_factor = 1.,
        **kwargs
    ):
        logits = self.forward(*args, p_uncond = 0., **kwargs)

        if guidance_factor == 0:
            return logits

        null_logits = self.forward(*args, p_uncond = 1., **kwargs)
        guidance_weighted_logits = (1 + guidance_factor)*logits - guidance_factor*null_logits
        return guidance_weighted_logits

    def predict(self, *args,
                guidance_factor=6.0,
                classes=None,
                **kwargs):
        if self.class_free_guidance:
            classes = default(classes,  torch.randint(0, self.num_classes, (8,)).cuda())
            kwargs = {**kwargs, "classes": classes}
            return self.forward_with_guidance(*args, guidance_factor=guidance_factor, **kwargs)
        else:
            return self.forward(*args, **kwargs)