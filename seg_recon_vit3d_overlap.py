import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import VitAttentionConfig, ViT_Attention
import math

# Embedding module that splits the input image into patches and embeds them
class Embedd(nn.Module):
    def __init__(self, in_channel=31, patch_size=32, embed_dim=768, overlap=False):
        super().__init__()
        self.in_channel = in_channel                # Number of input channels
        self.embed_dim = embed_dim                  # Embedding dimension for each patch
        self.patch_size = patch_size                # Size of each patch
        self.stride = patch_size // 2 if overlap else patch_size  # Stride = patch size / 2 for 50% overlap

        # Convolution to split the image into overlapping/non-overlapping patches and embed them
        self.patch_embedding_func = nn.Conv2d(
            in_channels=self.in_channel,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride
        )

        # Positional embedding will be initialized dynamically based on input size
        self.register_buffer('position_embedding', None, persistent=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # Apply patch embedding via Conv2D -> shape: (B, embed_dim, H', W')
        patches = self.patch_embedding_func(tensor)
        B, C, H, W = patches.shape
        num_patches = H * W

        # Flatten to (B, T, E) where T = H' * W'
        patches = patches.flatten(2).transpose(1, 2)

        # Lazily initialize positional embedding if needed
        if self.position_embedding is None or self.position_embedding.num_embeddings != num_patches:
            self.position_embedding = nn.Embedding(num_patches, self.embed_dim).to(tensor.device)

        # Create positional indices for each patch
        position_ids = torch.arange(num_patches, device=tensor.device).unsqueeze(0)

        # Add positional encoding to the patch embeddings
        embeddings = patches + self.position_embedding(position_ids)
        return embeddings, H, W


# Encoder = embedding + attention, embedding inputs B,C,Y,X and outputs (B, T, E), H, W, with h*w = T, attention inputs (B, T, E) and outputs B, T, E
class Encoder(nn.Module): 
    def __init__(self, config, in_channel=31, patch_size=32, embed_dim=768, overlap=False):
        super().__init__()
        self.embedding_function = Embedd(in_channel, patch_size, embed_dim, overlap) 
        self.attention = ViT_Attention(config)

    def forward(self, tensor: torch.Tensor):
        # Embed patches with position encoding
        embeddings, h, w = self.embedding_function(tensor)  # (B, T, E), h*w = T
        # Apply self-attention on the patches
        x = self.attention(embeddings)  # (B, T, E)
        return x, h, w


# Decoder that upsamples the encoded patch features back to a full-resolution image
class RecDecoder(nn.Module):
    def __init__(self, embed_dim, tot_channels, patch_size=32, overlap=False):
        super().__init__()

        # Compute required upsampling factor to restore original resolution
        self.stride = patch_size // 2 if overlap else patch_size
        self.upsample_factor = self.stride
        self.num_upsample_layers = int(math.log2(self.upsample_factor))

        channels = [embed_dim]
        for _ in range(self.num_upsample_layers):
            channels.append(max(channels[-1] // 2, tot_channels))
        
        self.dec_channels = channels
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.ConvTranspose2d(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=5,
                stride=2,
                padding=1
            ))
            layers.append(nn.BatchNorm2d(channels[i+1]))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(
            in_channels=channels[-1],
            out_channels=tot_channels,
            kernel_size=2,
            padding=1
        ))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x, h, w):
        B, T, E = x.shape
        x = x.transpose(1, 2).contiguous().view(B, E, h, w)
        return self.decoder(x)


# Main Vision Transformer-based segmentation/reconstruction model
class SegRecon_ViT_3D_Overlap(nn.Module):
    def __init__(self, C_input=31, total_channels=61, patch_size=32, emb_size=768, num_heads=12, ifft=False, overlap=False, nconv = 1):
        super().__init__()

        self.total_channels = total_channels
        self.C_input = C_input
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.ifft = ifft
        self.overlap = overlap
        self.nconv = nconv

        config = VitAttentionConfig(
            attention_dropout=0.0,
            num_attention_heads=num_heads,
            hidden_size=emb_size)

        self.encoder = Encoder(config, C_input, patch_size, emb_size, overlap)
        self.decoder = RecDecoder(embed_dim=emb_size, tot_channels=total_channels, patch_size=patch_size, overlap=overlap)
        print(self.decoder)

        self.out_conv = nn.Conv2d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False
        )

    def forward(self, x_input: torch.Tensor):
        B, C, Y, X = x_input.shape
        assert C == self.C_input, f"Expected {self.C_input} input channels, got {C}"

        stride = self.patch_size // 2 if self.overlap else self.patch_size
        if Y % stride != 0 or X % stride != 0:
            raise ValueError(f"Input dimensions ({Y}, {X}) must be divisible by stride {stride}.")

        x, h, w = self.encoder(x_input)
        x = self.decoder(x, h, w)
        if self.nconv ==1: 
            x = self.out_conv(x)
            
        else: 
            for i in range(self.nconv): 
                x = self.out_conv(x)

        if not self.ifft:
            x = torch.clip(x, 0, 1)

        return x
