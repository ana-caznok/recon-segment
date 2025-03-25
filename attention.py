
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

class Head(nn.Module):
    """A single attention head from multi-head attention."""

    def __init__(self, n_in, n_head, context_length):
        """
            n_in (int): Input embedding dimension.
            n_head (int): Output dimension of this head (head size).
            context_length (int): Not used here but may be relevant for masking later.
        """
        super().__init__()

        self.head_size = n_head # Store the output size for this head
        self.key = nn.Linear(n_in, n_head, bias=False)  # Linear projection to generate keys from input, no bias
        self.query = nn.Linear(n_in, n_head, bias=False)  # Linear projection to generate queries from input
        self.value = nn.Linear(n_in, n_head, bias=False) # Linear projection to generate values from input

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, T, E) B = batch size, T = sequence length, E = input dim
        Returns:
            Tensor: Attention output of shape (B, T, head_size)
        """
        B, T, E = x.shape  # Get dimensions

        # Project input x to key, query, and value vectors
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Compute scaled dot-product attention weights:
        wei = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size)) # q @ k^T gives attention scores, shape (B, T, T) , scale by sqrt(head_size) for numerical stability

        # Apply softmax over the last dimension to get attention probabilities
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # Multiply attention weights by values
        out = wei @ v  # (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, num_head, n_in, head_size, context_length):
        """
        Args:
            num_head (int): Number of attention heads.
            n_in (int): Input embedding dimension.
            head_size (int): Output dimension of each head.
            context_length (int): May be useful for future masking support.
        """
        super().__init__()

        self.head_size = head_size
        self.num_head = num_head

        # Create multiple attention heads using a ModuleList so PyTorch tracks them
        self.heads = nn.ModuleList([
            Head(n_in, head_size, context_length) for _ in range(num_head)
        ])

        # Final linear projection to project concatenated heads back to original input dim
        self.proj = nn.Linear(num_head * head_size, n_in)

    def forward(self, x):
        """
        Args: x: Input tensor of shape (B, T, E)
        Returns: Output tensor of shape (B, T, E)
        """
        # Run input through all heads independently and collect outputs
        out = [h(x) for h in self.heads]  # List of (B, T, head_size)

        # Concatenate along the embedding dimension (last dim)
        out = torch.cat(out, dim=-1)  # Shape: (B, T, num_head * head_size)

        # Project concatenated output back to input embedding dimension
        out = self.proj(out)  # Shape: (B, T, n_in)

        return out


# Configuration class to hold hyperparameters for the vision transformer model
@dataclass
class VitAttentionConfig:
    num_channels: int = 31                  # Number of input channels
    num_attention_heads: int = 12          # Number of attention heads
    hidden_size: int = 768                 # Dimensionality of patch embeddings
    attention_dropout: float = 0.0         # Dropout probability for attention weights

# Multi-head self-attention module used in Vision Transformers
class ViT_Attention(nn.Module):
    def __init__(self, config: VitAttentionConfig):
        super().__init__()
        
        # Store the config for access later if needed
        self.config = config

        # Embedding dimension for input and output
        self.embed_dim = config.hidden_size

        # Number of heads for multi-head attention
        self.num_heads = config.num_attention_heads

        # Dropout probability during training
        self.dropout = config.attention_dropout

        # Linear layer to project input embeddings into query space
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Linear layer to project input embeddings into key space
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Linear layer to project input embeddings into value space
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Final linear projection after attention outputs are combined
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states, h = None , w = None ):
        """
        Args:
            hidden_states: Input tensor of shape (B, T, E), where
                B = batch size,
                T = number of patches (tokens),
                E = embedding dimension (should match embed_dim)
        Returns:
            Tensor of shape (B, T, E), same shape as input but transformed via attention
        """
        # Unpack shape
        B, T, E = hidden_states.shape

        # Project input to query, key, and value spaces using respective linear layers
        q_states = self.q_proj(hidden_states)  # (B, T, E)
        k_states = self.k_proj(hidden_states)  # (B, T, E)
        v_states = self.v_proj(hidden_states)  # (B, T, E)

        # Reshape into multi-head format: (B, num_heads, T, head_dim), where head_dim = E // num_heads
        q_states = q_states.view(B, T, self.num_heads, E // self.num_heads).transpose(1, 2)
        k_states = k_states.view(B, T, self.num_heads, E // self.num_heads).transpose(1, 2)
        v_states = v_states.view(B, T, self.num_heads, E // self.num_heads).transpose(1, 2)

        # Compute attention scores: (B, num_heads, T, T)
        # Dot-product between query and key, scaled by sqrt(head_dim)
        attn_weights = (q_states @ k_states.transpose(-2, -1)) * (1.0 / math.sqrt(k_states.size(-1))) 

        # Normalize attention scores to probabilities using softmax
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Ensure data type consistency (especially if using mixed precision training)
        attn_weights = attn_weights.to(q_states.dtype)

        # Apply dropout to attention weights during training for regularization
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Compute weighted sum of values: (B, num_heads, T, head_dim)
        attn_outs = attn_weights @ v_states

        # Rearrange dimensions to go back to (B, T, E)
        attn_outs = attn_outs.transpose(1, 2)  # (B, T, num_heads, head_dim)
        attn_outs = attn_outs.reshape(B, T, E).contiguous()  # Flatten head dim back into E

        # Apply final linear projection
        attn_outs = self.out_proj(attn_outs)  # (B, T, E)

        return attn_outs

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm, reduce, dim):
        super().__init__()
        
        if norm == "group":
            norms = [nn.GroupNorm(num_groups=8, num_channels=out_ch) for _ in range(2)]
        elif norm == "instance":
            norms = [getattr(nn, f"InstanceNorm{dim}")(out_ch) for _ in range(2)]
        elif norm:
            norms = [getattr(nn, f"BatchNorm{dim}")(out_ch) for _ in range(2)]
        else:
            norms = [nn.Identity() for _ in range(2)]

        # Default conv block changed for separable convolutions
        self.conv = nn.Sequential(
            # SeparableConv(in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=False),
            getattr(nn, f"Conv{dim}")(in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=False),
            norms[0],
            nn.LeakyReLU(inplace=True),
            # SeparableConv(out_ch, out_ch, kernel_size=3, padding=1, stride=2 if reduce else 1, bias=False),
            getattr(nn, f"Conv{dim}")(out_ch, out_ch, kernel_size=3, padding=1, stride=2 if reduce else 1, bias=False),
            norms[1],
            nn.LeakyReLU(inplace=True)
        )

        self.residual_connection = getattr(nn, f"Conv{dim}")(in_ch, out_ch, kernel_size=1, padding=0, stride=2 if reduce else 1, bias=False)

    def forward(self, x):
        y = self.conv(x)
        return y + self.residual_connection(x)