import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from attention import VitAttentionConfig, ViT_Attention, DoubleConv

# Utility class for downsampling a 4D tensor by a fixed factor (unused in this version)
class DownsampleBatch():
    def __init__(self, factor: int):
        self.factor = factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, ::self.factor, ::self.factor]

    def __str__(self) -> str:
        return f"Downsample by a factor of {self.factor}, assuming 4D batch input"

# Embedding module that splits the input image into patches and embeds them
class Embedd(nn.Module):
    def __init__(self, in_channel=31, patch_size=32, embed_dim=768):
        super().__init__()
        self.in_channel = in_channel                # Number of input channels
        self.embed_dim = embed_dim                  # Embedding dimension for each patch
        self.patch_size = patch_size                # Size of each patch

        # Convolution to split the image into non-overlapping patches and embed them
        self.patch_embedding_func = nn.Conv2d(
            in_channels=self.in_channel,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Positional embedding will be initialized dynamically based on input size
        #self.position_embedding = None
        self.register_buffer('position_embedding', None, persistent=False)


    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # Apply patch embedding via Conv2D -> shape: (B, embed_dim, H', W')
        patches = self.patch_embedding_func(tensor)
        B, C, H, W = patches.shape
        num_patches = H * W

        # Flatten to (B, T, E) where T = H' * W'
        patches = patches.flatten(2).transpose(1, 2)

        # Lazily initialize positional embedding if needed : probably remove the if and leve the statement 
        if self.position_embedding is None or self.position_embedding.num_embeddings != num_patches:
            self.position_embedding = nn.Embedding(num_patches, self.embed_dim).to(tensor.device)

        # Create positional indices for each patch
        position_ids = torch.arange(num_patches, device=tensor.device).unsqueeze(0)

        # Add positional encoding to the patch embeddings
        embeddings = patches + self.position_embedding(position_ids)
        return embeddings, H, W
    

class Encoder(nn.Module): 
    def __init__(self, config, in_channel=31, patch_size=32, embed_dim=768):
        super().__init__()
        self.embedding_function = Embedd(in_channel, patch_size, embed_dim) 
        self.attention = ViT_Attention(config)

    def forward(self, tensor: torch.Tensor):

        # Embed patches with position encoding
        embeddings, h, w = self.embedding_function(tensor)  # (B, T, E), h*w = T
        # Apply self-attention on the patches
        x = self.attention(embeddings)  # (B, T, E)
        return x, h, w

        

# Decoder that upsamples the encoded patch features back to a full-resolution image
class RecDecoder(nn.Module):
    def __init__(self, embed_dim, tot_channels, patch_size = 32,upsampling_channels=[512, 256, 128, 64]):
        super().__init__()
        #if patch_size !=32: 
        #    factor = 32/patch_size
        #    for i in range(len(upsampling_channels)): 
        #        upsampling_channels[i] = int(upsampling_channels[i]*factor)

        print(upsampling_channels)
        # Define intermediate channel sizes for each transposed conv layer
        channels = [embed_dim] + upsampling_channels
        layers = []

        # Build upsampling layers using ConvTranspose2D
        for i in range(len(channels) - 1):
            layers.append(nn.ConvTranspose2d(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=4,
                stride=2,
                padding=1
            ))
            
            layers.append(nn.BatchNorm2d(channels[i+1]))  # Batch normalization
            layers.append(nn.ReLU(inplace=True))          # Activation
        
        # Final layer to map to target number of output channels
        layers.append(nn.Conv2d(
            in_channels=channels[-1],
            out_channels=tot_channels,
            kernel_size=3,
            padding=1
        ))

        factor = patch_size//16 #to output the right dimensions

        layers.append(nn.Upsample(scale_factor=factor, mode="bilinear"))
        # New layer that does not change output shape
        layers.append(nn.Conv2d(
            in_channels=tot_channels,
            out_channels=tot_channels,
            kernel_size=3,
            padding=1
        ))
        layers.append(nn.ReLU(inplace=True))

        # Store the full decoder as a Sequential module
        self.decoder = nn.Sequential(*layers)

    def forward(self, x, h, w):
        # Reshape from (B, T, E) -> (B, E, H', W') for 2D convolution
        B, T, E = x.shape
        x = x.transpose(1, 2).contiguous().view(B, E, h, w)

        # Decode to final image
        x = self.decoder(x)
        return x


# Main Vision Transformer-based segmentation/reconstruction model
class SegRecon_ViT_3D(nn.Module):
    def __init__(self, C_input=31, total_channels=61, patch_size=32, emb_size=768, num_heads=12,ifft=False):
        super().__init__()

        # Store model configuration
        self.total_channels = total_channels
        self.C_input = C_input
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.ifft = ifft

        # Attention configuration for the transformer block
        config = VitAttentionConfig(
            attention_dropout=0.0,
            num_attention_heads=num_heads,
            hidden_size=emb_size)

        # Vision Transformer attention block
        # Encoder = embedding + attention, embedding inputs B,C,Y,X and outputs (B, T, E), H, W, with h*w = T, attention inputs (B, T, E) and outputs B, T, E
        self.encoder = Encoder(config,C_input, patch_size, emb_size)

        # Decoder to reconstruct full image from patch embeddings
        self.decoder = RecDecoder(embed_dim=emb_size, tot_channels=total_channels, patch_size=patch_size)
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
        x,h,w = self.encoder(x_input) # outputs (B, T, E), H, W, with h*w = T, H is the number of vertical and W of horizontal patches 

        # Decode to full-resolution image
        x = self.decoder(x, h, w)  # outputs (B, total_channels, Y, X)

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

    model = SegRecon_ViT_3D()
    torchinfo.summary(model, input_size=(1, 31, 512, 512), device="cpu")
#layers.append(nn.Upsample(scale_factor=2, mode="bilinear"))