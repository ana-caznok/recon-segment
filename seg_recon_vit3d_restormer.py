import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ----------------------------
# PixelUnshuffle Layer
# ----------------------------
class PixelUnShuffle(nn.Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.r = downscale_factor

    def forward(self, x):
        b, c, h, w = x.size()
        r = self.r
        assert h % r == 0 and w % r == 0, "Input size must be divisible by patch size"
        out_c = c * (r ** 2)
        out_h = h // r
        out_w = w // r
        x = x.view(b, c, out_h, r, out_w, r)
        x = x.permute(0, 1, 3, 5, 2, 4).reshape(b, out_c, out_h, out_w)
        return x


# ----------------------------
# Restormer-style Attention (MDTA)
# ----------------------------
class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        k = rearrange(k, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        v = rearrange(v, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b h c (h1 w1) -> b (h c) h1 w1', h1=H, w1=W, h=self.num_heads)
        return self.proj(out)


# ----------------------------
# Patch Embedding via PixelUnshuffle
# ----------------------------
class PixelUnshuffleEmbed(nn.Module):
    def __init__(self, in_channels=31, patch_size=16, embed_dim=768):
        super().__init__()
        self.unshuffle = PixelUnShuffle(patch_size)
        self.proj = nn.Conv2d(in_channels * patch_size ** 2, embed_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.unshuffle(x)  # (B, C*p^2, H/p, W/p)
        x = self.proj(x)       # (B, embed_dim, H/p, W/p)
        return x


# ----------------------------
# Encoder Module
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=31, patch_size=16, embed_dim=768, num_heads=12):
        super().__init__()
        self.embed = PixelUnshuffleEmbed(in_channels, patch_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = MDTA(embed_dim, num_heads)

    def forward(self, x):
        feat = self.embed(x)  # (B, E, h, w)
        B, E, h, w = feat.shape

        feat_seq = feat.flatten(2).transpose(1, 2)
        feat_norm = self.norm(feat_seq).transpose(1, 2).reshape(B, E, h, w)

        x_attn = self.attn(feat_norm)
        return x_attn, h, w


# ----------------------------
# Decoder Module
# ----------------------------
class RecDecoder(nn.Module):
    def __init__(self, embed_dim, out_channels, patch_size, up_channels=[512, 256, 128, 64]):
        super().__init__()
        num_upsamples = int(torch.log2(torch.tensor(patch_size)).item())
        chs = [embed_dim] + up_channels[:num_upsamples]
        layers = []

        for i in range(len(chs) - 1):
            layers += [
                nn.ConvTranspose2d(chs[i], chs[i+1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(chs[i+1]),
                nn.ReLU(inplace=True)
            ]

        layers += [
            nn.Conv2d(chs[-1], out_channels, kernel_size=3, padding=1)
        ]

        self.decoder = nn.Sequential(*layers)

    def forward(self, x, h, w):
        B, T, E = x.shape
        x = x.transpose(1, 2).reshape(B, E, h, w)
        return self.decoder(x)


# ----------------------------
# SegRecon_ViT_3D_Rest (Main Model)
# ----------------------------
class SegRecon_ViT_3D_Rest(nn.Module):
    def __init__(self, C_input=31, total_channels=61, patch_size=16, emb_size=768, num_heads=12, ifft=False):
        super().__init__()
        assert patch_size in [4, 8, 16, 32], "Patch size must be power of 2 and divisible by input dimensions"
        self.C_input = C_input
        self.total_channels = total_channels
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.ifft = ifft

        self.encoder = Encoder(C_input, patch_size, emb_size, num_heads)
        self.decoder = RecDecoder(emb_size, total_channels, patch_size)
        self.out_conv = nn.Conv2d(total_channels, total_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.C_input
        assert H % self.patch_size == 0 and W % self.patch_size == 0

        enc_feat, h, w = self.encoder(x)
        seq = enc_feat.flatten(2).transpose(1, 2)  # (B, T, E)
        out = self.decoder(seq, h, w)
        out = self.out_conv(out)

        if not self.ifft:
            out = torch.clamp(out, 0, 1)
        return out


# ----------------------------
# Test Forward Pass
# ----------------------------
if __name__ == "__main__":
    model = SegRecon_ViT_3D_Rest(C_input=31, total_channels=61, patch_size=16, emb_size=384, num_heads=6)
    dummy = torch.randn(1, 31, 256, 256)
    out = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")
