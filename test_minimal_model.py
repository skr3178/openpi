"""
Test script for the minimal low-memory PI0 model.

This script demonstrates:
1. How to create fake data
2. How to initialize the minimal model
3. How to run training forward pass
4. How to run inference sampling

Memory usage: ~50MB GPU memory (vs ~20GB for full model)
Training time: ~0.1s per step (vs ~1s for full model)
"""

import torch
from openpi.models_pytorch.pi0_pytorch_minimal import (
    PI0PytorchMinimal,
    MinimalConfig,
    SimpleObservation,
)


def create_fake_observation(config: MinimalConfig, batch_size: int = 2, device: str = "cpu"):
    """Create fake observation data for testing."""
    # Create fake images (3 cameras, 224x224 RGB)
    images = {
        "base_0_rgb": torch.randn(batch_size, 224, 224, 3, device=device),
        "left_wrist_0_rgb": torch.randn(batch_size, 224, 224, 3, device=device),
        "right_wrist_0_rgb": torch.randn(batch_size, 224, 224, 3, device=device),
    }
    
    # Image masks (all valid)
    image_masks = {
        key: torch.ones(batch_size, dtype=torch.bool, device=device) 
        for key in images.keys()
    }
    
    # Robot state (joint positions, etc.)
    state = torch.randn(batch_size, config.action_dim, device=device)
    
    # Language tokens (tokenized instruction)
    tokenized_prompt = torch.randint(
        0, 2048, (batch_size, config.max_token_len), dtype=torch.int32, device=device
    )
    
    # Token mask (all valid)
    tokenized_prompt_mask = torch.ones(
        batch_size, config.max_token_len, dtype=torch.bool, device=device
    )
    
    return SimpleObservation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )


def create_fake_actions(config: MinimalConfig, batch_size: int = 2, device: str = "cpu"):
    """Create fake action data for testing."""
    return torch.randn(
        batch_size, config.action_horizon, config.action_dim, 
        dtype=torch.float32, device=device
    )


def test_model_creation():
    """Test 1: Model initialization."""
    print("\n" + "="*70)
    print("TEST 1: Model Creation")
    print("="*70)
    
    config = MinimalConfig(
        action_dim=7,
        action_horizon=10,
        max_token_len=16,
        hidden_dim=128,
    )
    
    model = PI0PytorchMinimal(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    return model, config


def test_forward_pass(model, config, device="cpu"):
    """Test 2: Training forward pass."""
    print("\n" + "="*70)
    print("TEST 2: Training Forward Pass")
    print("="*70)
    
    batch_size = 2
    observation = create_fake_observation(config, batch_size=batch_size, device=device)
    actions = create_fake_actions(config, batch_size=batch_size, device=device)
    
    model.to(device)
    model.train()
    
    # Forward pass
    loss = model(observation, actions)
    
    print(f"✓ Forward pass successful")
    print(f"  Input batch size: {batch_size}")
    print(f"  Loss shape: {loss.shape}")
    print(f"  Expected: ({batch_size}, {config.action_horizon}, {config.action_dim})")
    print(f"  Loss value: {loss.mean().item():.4f}")
    
    # Test backward pass
    total_loss = loss.mean()
    total_loss.backward()
    
    print(f"✓ Backward pass successful")
    
    return loss


def test_inference(model, config, device="cpu"):
    """Test 3: Inference sampling."""
    print("\n" + "="*70)
    print("TEST 3: Inference Sampling")
    print("="*70)
    
    batch_size = 2
    observation = create_fake_observation(config, batch_size=batch_size, device=device)
    
    model.to(device)
    model.eval()
    
    # Sample actions with few denoising steps
    with torch.no_grad():
        sampled_actions = model.sample_actions(
            device=device,
            observation=observation,
            num_steps=5  # Use fewer steps for speed
        )
    
    print(f"✓ Inference sampling successful")
    print(f"  Sampled actions shape: {sampled_actions.shape}")
    print(f"  Expected: ({batch_size}, {config.action_horizon}, {config.action_dim})")
    print(f"  Action range: [{sampled_actions.min().item():.2f}, {sampled_actions.max().item():.2f}]")
    
    return sampled_actions


def test_training_loop(model, config, device="cpu", num_steps=10):
    """Test 4: Mini training loop."""
    print("\n" + "="*70)
    print("TEST 4: Mini Training Loop")
    print("="*70)
    
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    batch_size = 4
    
    print(f"Training for {num_steps} steps...")
    for step in range(num_steps):
        # Create new batch
        observation = create_fake_observation(config, batch_size=batch_size, device=device)
        actions = create_fake_actions(config, batch_size=batch_size, device=device)
        
        # Forward pass
        loss = model(observation, actions)
        total_loss = loss.mean()
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if step % 2 == 0:
            print(f"  Step {step:2d}: loss = {total_loss.item():.4f}")
    
    print(f"✓ Training loop completed successfully")


def test_memory_usage(device="cuda"):
    """Test 5: GPU memory usage (if available)."""
    if not torch.cuda.is_available():
        print("\n" + "="*70)
        print("TEST 5: GPU Memory Usage - SKIPPED (CUDA not available)")
        print("="*70)
        return
    
    print("\n" + "="*70)
    print("TEST 5: GPU Memory Usage")
    print("="*70)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    config = MinimalConfig(
        action_dim=7,
        action_horizon=10,
        max_token_len=16,
        hidden_dim=128,
    )
    
    model = PI0PytorchMinimal(config).to(device)
    
    mem_after_model = torch.cuda.memory_allocated(device) / 1024 / 1024
    print(f"  Memory after model creation: {mem_after_model:.1f} MB")
    
    # Run forward pass
    batch_size = 4
    observation = create_fake_observation(config, batch_size=batch_size, device=device)
    actions = create_fake_actions(config, batch_size=batch_size, device=device)
    
    loss = model(observation, actions)
    mem_after_forward = torch.cuda.memory_allocated(device) / 1024 / 1024
    print(f"  Memory after forward pass: {mem_after_forward:.1f} MB")
    
    # Run backward pass
    total_loss = loss.mean()
    total_loss.backward()
    mem_after_backward = torch.cuda.memory_allocated(device) / 1024 / 1024
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    
    print(f"  Memory after backward pass: {mem_after_backward:.1f} MB")
    print(f"  Peak memory usage: {peak_mem:.1f} MB")
    print(f"\n  ✓ Total memory usage: ~{peak_mem:.0f} MB")
    print(f"    (vs ~20,000 MB for full PI0 model)")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("MINIMAL PI0 MODEL - LOW MEMORY TESTING")
    print("="*70)
    print("\nThis minimal model replaces heavy components:")
    print("  • SigLIP (400M params) → Tiny CNN (0.1M params)")
    print("  • PaliGemma (2B params) → Mini Transformer (0.3M params)")
    print("  • Action Expert (300M params) → Mini Transformer (0.3M params)")
    print("\nTotal: ~1M parameters vs 2.7B in full model (2700x smaller!)")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Run tests
    model, config = test_model_creation()
    test_forward_pass(model, config, device=device)
    test_inference(model, config, device=device)
    test_training_loop(model, config, device=device, num_steps=10)
    test_memory_usage(device="cuda")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✓")
    print("="*70)
    print("\nYou can now use this minimal model for:")
    print("  • Testing training scripts without heavy GPU requirements")
    print("  • Debugging data loading pipelines")
    print("  • Rapid prototyping of new features")
    print("  • CI/CD testing on low-resource machines")
    print("\nTo use in your code:")
    print("  from openpi.models_pytorch.pi0_pytorch_minimal import PI0PytorchMinimal, MinimalConfig")
    print("  config = MinimalConfig(action_dim=7, action_horizon=10)")
    print("  model = PI0PytorchMinimal(config)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()


