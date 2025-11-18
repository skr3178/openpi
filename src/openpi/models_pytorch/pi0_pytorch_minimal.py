"""
Minimal low-memory version of PI0Pytorch for testing.
This version replaces the heavy PaliGemma and Gemma models with tiny transformers.

Usage:
    from openpi.models_pytorch.pi0_pytorch_minimal import PI0PytorchMinimal, MinimalConfig
    
    config = MinimalConfig(
        action_dim=7,
        action_horizon=10,
        max_token_len=16,
    )
    model = PI0PytorchMinimal(config)
"""

import dataclasses
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ============================================================================
# MINIMAL TRANSFORMER COMPONENTS
# ============================================================================


class MinimalAttention(nn.Module):
    """Lightweight attention mechanism."""
    
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn = attn + attention_mask[:, None, :, :]
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MinimalMLP(nn.Module):
    """Lightweight MLP."""
    
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class MinimalTransformerBlock(nn.Module):
    """Lightweight transformer block."""
    
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MinimalAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MinimalMLP(dim)
    
    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class MinimalVisionEncoder(nn.Module):
    """Tiny vision encoder - replaces SigLIP."""
    
    def __init__(self, output_dim=128, patch_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(3, output_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        # x: [B, H, W, C] -> convert to [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)  # [B, dim, H//patch, W//patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, dim]
        x = self.norm(x)
        return x


class MinimalLanguageTransformer(nn.Module):
    """Tiny language transformer - replaces PaliGemma."""
    
    def __init__(self, vocab_size=2048, dim=128, depth=2, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            MinimalTransformerBlock(dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embedding(x)
        
        for block in self.blocks:
            x = block(x, attention_mask)
        
        x = self.norm(x)
        return x


class MinimalActionTransformer(nn.Module):
    """Tiny action transformer - replaces Action Expert Gemma."""
    
    def __init__(self, dim=128, depth=2, num_heads=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            MinimalTransformerBlock(dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, attention_mask=None):
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.norm(x)
        return x


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclasses.dataclass
class MinimalConfig:
    """Minimal configuration for low-memory testing."""
    action_dim: int = 7
    action_horizon: int = 10
    max_token_len: int = 16
    hidden_dim: int = 128
    vision_patch_size: int = 32
    lang_depth: int = 2
    action_depth: int = 2
    num_heads: int = 4
    dtype: str = "float32"


# ============================================================================
# MINIMAL PI0 MODEL
# ============================================================================


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
    
    dtype = torch.float32 if device.type == "cpu" else torch.float32
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(pad_masks, att_masks):
    """Create 2D attention masks from padding and attention masks."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0PytorchMinimal(nn.Module):
    """
    Minimal low-memory version of PI0 for testing.
    
    Key differences from full PI0:
    - Uses tiny transformers (~1M params total vs 2.3B)
    - Simplified vision encoder (conv-based vs SigLIP)
    - Minimal transformer depths (2 layers vs 18+)
    - Small hidden dimension (128 vs 1024-2048)
    - No gradient checkpointing needed
    - No KV-cache needed
    """
    
    def __init__(self, config: MinimalConfig):
        super().__init__()
        self.config = config
        
        # Vision encoder (replaces SigLIP)
        self.vision_encoder = MinimalVisionEncoder(
            output_dim=config.hidden_dim,
            patch_size=config.vision_patch_size
        )
        
        # Language transformer (replaces PaliGemma)
        self.lang_transformer = MinimalLanguageTransformer(
            vocab_size=2048,
            dim=config.hidden_dim,
            depth=config.lang_depth,
            num_heads=config.num_heads
        )
        
        # Action transformer (replaces Action Expert)
        self.action_transformer = MinimalActionTransformer(
            dim=config.hidden_dim,
            depth=config.action_depth,
            num_heads=config.num_heads
        )
        
        # Action projections
        self.action_in_proj = nn.Linear(config.action_dim, config.hidden_dim)
        self.action_out_proj = nn.Linear(config.hidden_dim, config.action_dim)
        
        # State projection
        self.state_proj = nn.Linear(config.action_dim, config.hidden_dim)
        
        # Time conditioning MLPs
        self.action_time_mlp_in = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
        self.action_time_mlp_out = nn.Linear(config.hidden_dim, config.hidden_dim)
    
    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Convert 2D masks to 4D format for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -1e9)
    
    def embed_prefix(self, images_list, img_masks_list, lang_tokens, lang_masks):
        """Embed images and language tokens."""
        embs = []
        pad_masks = []
        att_masks = []
        
        # Process images
        for img, img_mask in zip(images_list, img_masks_list, strict=True):
            img_emb = self.vision_encoder(img)
            bsize, num_patches = img_emb.shape[:2]
            
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_patches))
            att_masks += [0] * num_patches
        
        # Process language tokens
        lang_emb = self.lang_transformer.embedding(lang_tokens)
        lang_emb = lang_emb * math.sqrt(self.config.hidden_dim)
        
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs
        
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        
        return embs, pad_masks, att_masks
    
    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy actions, and timestep."""
        embs = []
        pad_masks = []
        att_masks = []
        
        # Embed state
        state_emb = self.state_proj(state)
        embs.append(state_emb[:, None, :])
        
        bsize = state_emb.shape[0]
        device = state_emb.device
        
        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1]
        
        # Embed timestep
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.hidden_dim, min_period=4e-3, max_period=4.0, device=device
        )
        
        # Embed actions and fuse with time
        action_emb = self.action_in_proj(noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        
        # Apply MLP
        x = self.action_time_mlp_in(action_time_emb)
        x = F.silu(x)
        action_time_emb = self.action_time_mlp_out(x)
        
        embs.append(action_time_emb)
        
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)
        
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))
        
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        
        return embs, pad_masks, att_masks
    
    def sample_noise(self, shape, device):
        """Sample Gaussian noise."""
        return torch.randn(shape, dtype=torch.float32, device=device)
    
    def sample_time(self, bsize, device):
        """Sample timesteps from Beta distribution."""
        alpha_t = torch.as_tensor(1.5, dtype=torch.float32, device=device)
        beta_t = torch.as_tensor(1.0, dtype=torch.float32, device=device)
        dist = torch.distributions.Beta(alpha_t, beta_t)
        time_beta = dist.sample((bsize,))
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)
    
    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Training forward pass - compute flow-matching loss."""
        # Extract observation components
        images_list = list(observation.images.values())
        img_masks_list = list(observation.image_masks.values())
        lang_tokens = observation.tokenized_prompt
        lang_masks = observation.tokenized_prompt_mask
        state = observation.state
        
        # Sample noise and time
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        
        # Flow-matching: x_t = t*noise + (1-t)*action
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        
        # Embed prefix and suffix
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images_list, img_masks_list, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)
        
        # Concatenate
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        
        # Create 2D attention masks
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        
        # Process through transformers
        # Language transformer on prefix
        prefix_out = self.lang_transformer(None, att_2d_masks_4d[:, :, :prefix_embs.shape[1], :prefix_embs.shape[1]], prefix_embs)
        
        # Action transformer on full sequence
        all_embs = torch.cat([prefix_out, suffix_embs], dim=1)
        all_out = self.action_transformer(all_embs, att_2d_masks_4d)
        
        # Extract action outputs
        suffix_out = all_out[:, -self.config.action_horizon:]
        
        # Project to action space
        v_t = self.action_out_proj(suffix_out)
        
        # Compute loss
        return F.mse_loss(u_t, v_t, reduction="none")
    
    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Inference - iteratively denoise to get actions."""
        bsize = observation.state.shape[0]
        
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        
        # Extract observation
        images_list = list(observation.images.values())
        img_masks_list = list(observation.image_masks.values())
        lang_tokens = observation.tokenized_prompt
        lang_masks = observation.tokenized_prompt_mask
        state = observation.state
        
        # Embed prefix once
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images_list, img_masks_list, lang_tokens, lang_masks
        )
        
        # Iterative denoising
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            
            # Embed suffix for current noisy action
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, expanded_time)
            
            # Create attention masks
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
            
            # Forward pass
            prefix_out = self.lang_transformer(None, att_2d_masks_4d[:, :, :prefix_embs.shape[1], :prefix_embs.shape[1]], prefix_embs)
            all_embs = torch.cat([prefix_out, suffix_embs], dim=1)
            all_out = self.action_transformer(all_embs, att_2d_masks_4d)
            
            # Get velocity prediction
            suffix_out = all_out[:, -self.config.action_horizon:]
            v_t = self.action_out_proj(suffix_out)
            
            # Euler integration step
            x_t = x_t + dt * v_t
            time += dt
        
        return x_t


# ============================================================================
# HELPER CLASS FOR OBSERVATION
# ============================================================================


class SimpleObservation:
    """Simple observation class for testing."""
    
    def __init__(self, images, image_masks, state, tokenized_prompt, tokenized_prompt_mask):
        self.images = images
        self.image_masks = image_masks
        self.state = state
        self.tokenized_prompt = tokenized_prompt
        self.tokenized_prompt_mask = tokenized_prompt_mask


