import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from attention import VitAttentionConfig, ViT_Attention, DoubleConv

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class Encoder_ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()


    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        #x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = torch.einsum('bnd,bne->bnde', x, x)
        b,t,e1,e2 = x.shape
        #conv_mlp = nn.Conv2d(in_channels= t,
                                   # out_channels =t,
                                   # kernel_size =3,
                                   # stride = 1,
                                   # padding = 1)
        #x = conv_mlp(x)
        return x

import torch
import torch.nn as nn

# Optional helper function to compute output shape
def get_output_shape(layer_type, layer_params, input_shape):
    # Simulate a dummy tensor to pass through the layer
    dummy = torch.zeros(1, layer_params["in_channels"], *input_shape)
    layer = layer_type(**layer_params)
    output = layer(dummy)
    return output.shape[2:]  # Return H, W

class RecDecoder(nn.Module):
    def __init__(self, 
                 dim_head,           # dim_head + 1 or other components
                 tot_channels,       # final output image channels
                 emb_size,           # input spatial size (H=W)
                 img_size            # target output image size
                 ):
        super().__init__()

        # Initial input channel size (e.g., dim_head + 4 as given)
        t = dim_head + 4
        self.t = t

        # Store layers sequentially
        layers = []

        # First conv layer (preserves shape)
        layers.append(nn.Conv2d(in_channels=t,
                                out_channels=t - 1,
                                kernel_size=3,
                                stride=1,
                                padding=1))
        layers.append(nn.BatchNorm2d(t - 1))
        layers.append(nn.ReLU(inplace=True))

        # Keep track of shape after each operation
        current_shape = (emb_size, emb_size)

        # Conv2d: downsample
        conv1_params = {
            "in_channels": t - 1,
            "out_channels": t - 2,
            "kernel_size": 3,
            "stride": 2,
            "padding": 1
        }
        current_shape = get_output_shape(nn.Conv2d, conv1_params, current_shape)
        layers.append(nn.Conv2d(**conv1_params))
        layers.append(nn.BatchNorm2d(t - 2))
        layers.append(nn.ReLU(inplace=True))

        # Conv2d: further downsample
        conv2_params = {
            "in_channels": t - 2,
            "out_channels": t - 3,
            "kernel_size": 5,
            "stride": 3,
            "padding": 1
        }
        current_shape = get_output_shape(nn.Conv2d, conv2_params, current_shape)
        layers.append(nn.Conv2d(**conv2_params))
        layers.append(nn.BatchNorm2d(t - 3))
        layers.append(nn.ReLU(inplace=True))

        # ConvTranspose2d: upsample
        deconv_params = {
            "in_channels": t - 3,
            "out_channels": t - 4,
            "kernel_size": 4,
            "stride": 2,
            "padding": 1
        }
        current_shape = get_output_shape(nn.ConvTranspose2d, deconv_params, current_shape)
        layers.append(nn.ConvTranspose2d(**deconv_params))
        layers.append(nn.BatchNorm2d(t - 4))
        layers.append(nn.ReLU(inplace=True))

        # Final conv layer to refine and produce tot_channels
        layers.append(nn.Conv2d(in_channels=t - 4,
                                out_channels=tot_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1))
        layers.append(nn.ReLU(inplace=True))

        # Optional final upsample if spatial shape is still smaller than target
        if current_shape != (img_size, img_size):
            scale = img_size // current_shape[0]
            layers.append(nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False))

        # Store decoder as a sequential model
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: Tensor of shape (B, t, emb_size, emb_size)
        """
        return self.decoder(x)


# Main Vision Transformer-based segmentation/reconstruction model
class SegRecon_ViT_3D_2(nn.Module):
    def __init__(self, C_input=31, total_channels=61, patch_size=32, emb_size=768, num_heads=12,ifft=False, img_size=256):
        super().__init__()

        # Store model configuration
        self.total_channels = total_channels
        self.C_input = C_input
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.ifft = ifft
        self.img_size = img_size
        self.dim_head = total_channels + 4
       
        # Vision Transformer attention block
        # Encoder = embedding + attention, embedding inputs B,C,Y,X and outputs (B, T, E), H, W, with h*w = T, attention inputs (B, T, E) and outputs B, T, E
        self.encoder = Encoder_ViT(image_size = self.img_size,
                                   patch_size= self.patch_size,
                                   dim= emb_size,
                                   depth=6,
                                   heads = self.num_heads,
                                   mlp_dim=self.emb_size,
                                   pool = 'cls',
                                   channels=self.C_input, 
                                   dim_head=self.dim_head, 
                                   dropout=0,
                                   emb_dropout=0)

        # Decoder to reconstruct full image from patch embeddings
        self.decoder = RecDecoder(dim_head=self.dim_head-4, 
                                  tot_channels=self.total_channels, 
                                  emb_size = self.emb_size,             # input spatial size
                                  img_size = self.img_size)
                                  
                                  
        print(self.decoder)

        # Final 1x1 convolution to project to desired channel count
        self.out_conv = nn.Conv2d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False
        )

    def forward(self, x_input: torch.Tensor):
        # Validate input shape
        B, C, Y, X = x_input.shape
        assert C == self.C_input, f"Expected {self.C_input} input channels, got {C}"

        # Ensure image dimensions are divisible by patch size
        if Y % self.patch_size != 0 or X % self.patch_size != 0:
            raise ValueError(f"Input dimensions ({Y}, {X}) must be divisible by patch size {self.patch_size}.")
        
        # Encoder: embedd image patches and apply self attention     
        x = self.encoder(x_input) # outputs (B, T, E), H, W, with h*w = T, H is the number of vertical and W of horizontal patches 

        # Decode to full-resolution image
        x = self.decoder(x)  # outputs (B, total_channels, Y, X)

        # Final channel projection
        x = self.out_conv(x)
        #print(x.size)
        if self.ifft == False: 
            # Clip output between 0 and 1
            x = torch.clip(x,0,1)

        return x


# Optional: summarize model architecture for debugging
if __name__ == "__main__":
    import torchinfo

    model = SegRecon_ViT_3D_2()
    torchinfo.summary(model, input_size=(1, 31, 256, 256), device="cpu")
#layers.append(nn.Upsample(scale_factor=2, mode="bilinear"))