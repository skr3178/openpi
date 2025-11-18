"""Tests for PI0Pytorch model."""

import pytest
import torch

from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch


def create_fake_observation(config: pi0_config.Pi0Config, batch_size: int = 2, device: str = "cpu"):
    """Create a fake observation object for testing."""
    image_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    
    # Create images: [B, H, W, C] format (224, 224, 3)
    images = {}
    for key in image_keys:
        images[key] = torch.randn(
            batch_size, 224, 224, 3, dtype=torch.float32, device=device
        )
    
    # Create image masks
    image_masks = {}
    for key in image_keys:
        image_masks[key] = torch.ones(batch_size, dtype=torch.bool, device=device)
    
    # Create state: [B, action_dim]
    state = torch.randn(batch_size, config.action_dim, dtype=torch.float32, device=device)
    
    # Create tokenized prompt: [B, max_token_len]
    tokenized_prompt = torch.randint(
        0, 2048, (batch_size, config.max_token_len), dtype=torch.int32, device=device
    )
    
    # Create tokenized prompt mask
    tokenized_prompt_mask = torch.ones(
        batch_size, config.max_token_len, dtype=torch.bool, device=device
    )
    
    # Create token_ar_mask and token_loss_mask (can be None for pi0, but set for compatibility)
    token_ar_mask = None
    token_loss_mask = None
    
    # Create a simple observation object
    class SimpleObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    return SimpleObservation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        token_ar_mask=token_ar_mask,
        token_loss_mask=token_loss_mask,
    )


def create_fake_actions(config: pi0_config.Pi0Config, batch_size: int = 2, device: str = "cpu"):
    """Create fake actions for testing."""
    return torch.randn(
        batch_size, config.action_horizon, config.action_dim, dtype=torch.float32, device=device
    )


@pytest.mark.parametrize("pi05", [False, True])
def test_model_initialization(pi05):
    """Test that the model can be initialized."""
    config = pi0_config.Pi0Config(
        action_dim=32,
        action_horizon=50,
        max_token_len=48 if not pi05 else 200,
        pi05=pi05,
        dtype="float32",  # Use float32 for CPU testing
    )
    
    model = pi0_pytorch.PI0Pytorch(config)
    model.eval()  # Set to eval mode
    
    assert model.config == config
    assert model.pi05 == pi05


def test_model_forward_pass():
    """Test the forward pass of the model."""
    config = pi0_config.Pi0Config(
        action_dim=32,
        action_horizon=50,
        max_token_len=48,
        pi05=False,
        dtype="float32",  # Use float32 for CPU testing
    )
    
    model = pi0_pytorch.PI0Pytorch(config)
    model.eval()
    
    batch_size = 2
    device = "cpu"
    observation = create_fake_observation(config, batch_size=batch_size, device=device)
    actions = create_fake_actions(config, batch_size=batch_size, device=device)
    
    # Test forward pass
    loss = model.forward(observation, actions)
    
    # Check loss shape: should be [batch_size, action_horizon, action_dim]
    assert loss.shape == (batch_size, config.action_horizon, config.action_dim)
    assert loss.dtype == torch.float32


def test_model_sample_actions():
    """Test the sample_actions method."""
    config = pi0_config.Pi0Config(
        action_dim=32,
        action_horizon=50,
        max_token_len=48,
        pi05=False,
        dtype="float32",  # Use float32 for CPU testing
    )
    
    model = pi0_pytorch.PI0Pytorch(config)
    model.eval()
    
    batch_size = 2
    device = "cpu"
    observation = create_fake_observation(config, batch_size=batch_size, device=device)
    
    # Test sample_actions
    with torch.no_grad():
        sampled_actions = model.sample_actions(device=device, observation=observation, num_steps=5)
    
    # Check output shape: should be [batch_size, action_horizon, action_dim]
    assert sampled_actions.shape == (batch_size, config.action_horizon, config.action_dim)
    assert sampled_actions.dtype == torch.float32


def test_model_gradient_checkpointing():
    """Test gradient checkpointing enable/disable."""
    config = pi0_config.Pi0Config(
        action_dim=32,
        action_horizon=50,
        max_token_len=48,
        pi05=False,
        dtype="float32",
    )
    
    model = pi0_pytorch.PI0Pytorch(config)
    
    # Initially should be disabled
    assert not model.is_gradient_checkpointing_enabled()
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    assert model.is_gradient_checkpointing_enabled()
    
    # Disable gradient checkpointing
    model.gradient_checkpointing_disable()
    assert not model.is_gradient_checkpointing_enabled()


def test_model_with_pi05():
    """Test model with pi05=True configuration."""
    config = pi0_config.Pi0Config(
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
        pi05=True,
        dtype="float32",
    )
    
    model = pi0_pytorch.PI0Pytorch(config)
    model.eval()
    
    batch_size = 2
    device = "cpu"
    observation = create_fake_observation(config, batch_size=batch_size, device=device)
    actions = create_fake_actions(config, batch_size=batch_size, device=device)
    
    # Test forward pass
    loss = model.forward(observation, actions)
    assert loss.shape == (batch_size, config.action_horizon, config.action_dim)
    
    # Test sample_actions
    with torch.no_grad():
        sampled_actions = model.sample_actions(device=device, observation=observation, num_steps=5)
    assert sampled_actions.shape == (batch_size, config.action_horizon, config.action_dim)


def test_helper_functions():
    """Test helper functions like create_sinusoidal_pos_embedding."""
    device = torch.device("cpu")
    time = torch.tensor([0.1, 0.5, 0.9], device=device)
    dimension = 64
    
    embedding = pi0_pytorch.create_sinusoidal_pos_embedding(
        time, dimension, min_period=4e-3, max_period=4.0, device=device
    )
    
    assert embedding.shape == (3, dimension)
    assert embedding.dtype == torch.float64  # Uses float64 internally


def test_make_att_2d_masks():
    """Test the make_att_2d_masks function."""
    batch_size = 2
    seq_len = 5
    
    pad_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
    att_masks = torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.bool).expand(batch_size, seq_len)
    
    att_2d_masks = pi0_pytorch.make_att_2d_masks(pad_masks, att_masks)
    
    assert att_2d_masks.shape == (batch_size, seq_len, seq_len)
    assert att_2d_masks.dtype == torch.bool


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_on_gpu():
    """Test model on GPU if available."""
    config = pi0_config.Pi0Config(
        action_dim=32,
        action_horizon=50,
        max_token_len=48,
        pi05=False,
        dtype="float32",
    )
    
    device = "cuda"
    model = pi0_pytorch.PI0Pytorch(config).to(device)
    model.eval()
    
    batch_size = 2
    observation = create_fake_observation(config, batch_size=batch_size, device=device)
    actions = create_fake_actions(config, batch_size=batch_size, device=device)
    
    # Test forward pass on GPU
    loss = model.forward(observation, actions)
    assert loss.shape == (batch_size, config.action_horizon, config.action_dim)
    assert loss.device.type == "cuda"

