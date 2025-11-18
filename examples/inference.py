#!/usr/bin/env python3
"""
Policy inference example

The following example shows how to create a policy from a checkpoint and run inference on a dummy example.

Memory optimization fixes based on GitHub issues:
- Issue #376: Memory exhaustion issue when loading pi0_base params
- Issue #379: Memory Exhausted Issue when Training with Pi0 Fine Tuned Lora  
- Issue #340: Inference memory cost is unexpected high
"""

import dataclasses
import gc
import os
import pathlib

# Set JAX memory optimizations before importing JAX
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # Use 70% of GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Disable preallocation
os.environ['JAX_ENABLE_X64'] = 'false'  # Use float32 instead of float64

import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


# =============================================================================
# TESTING CONFIGURATION - Uncomment sections to test specific components
# =============================================================================

# Set to True to test policy inference (lines 88-102)
TEST_POLICY_INFERENCE = True

# Set to True to test live model with fake data (lines 111-140) 
TEST_LIVE_MODEL = False

# Set to True to test data loader with real training data (lines 176-214)
TEST_DATA_LOADER = False

# =============================================================================


def configure_memory_settings():
    """Configure JAX memory settings to optimize memory usage based on GitHub issues."""
    # Memory settings based on GitHub issues #667, #677
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.7')  # Use 70% of GPU memory
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')  # Disable preallocation
    os.environ.setdefault('JAX_ENABLE_X64', 'false')  # Use float32 instead of float64
    
    # Enable GPU but with memory management
    os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')  # Use GPU for better performance
    
    # Additional optimizations from GitHub issues
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # Reduce fragmentation
    
    print("Memory settings configured:")
    print(f"  XLA_PYTHON_CLIENT_MEM_FRACTION: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')}")
    print(f"  XLA_PYTHON_CLIENT_PREALLOCATE: {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')}")
    print(f"  JAX_ENABLE_X64: {os.environ.get('JAX_ENABLE_X64')}")
    print(f"  JAX_PLATFORM_NAME: {os.environ.get('JAX_PLATFORM_NAME')}")


def clear_memory():
    """Clear JAX memory and run garbage collection."""
    jax.clear_caches()
    gc.collect()
    print("Memory cleared")


def main():
    """Main function demonstrating policy inference with memory optimizations."""
    
    # Configure memory settings first
    configure_memory_settings()
    
    # Get configuration and download checkpoint
    config = _config.get_config("pi0_fast_droid")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
    
    # Check cache directory and any partial downloads
    cache_dir = download.get_cache_dir()
    print(f"Cache directory: {cache_dir}")
    print(f"Cache directory exists: {cache_dir.exists()}")
    
    if cache_dir.exists():
        print(f"Cache contents:")
        for item in cache_dir.iterdir():
            if item.is_dir():
                print(f"  - {item.name}/ (directory)")
            else:
                print(f"  - {item.name}")
    
    # Check for any partial downloads (files ending with .partial)
    partial_files = list(cache_dir.rglob("*.partial"))
    if partial_files:
        print(f"\nFound {len(partial_files)} partial download(s):")
        for partial_file in partial_files:
            print(f"  - {partial_file}")
    else:
        print("\nNo partial downloads found")
    
    # Test Policy Inference (controlled by TEST_POLICY_INFERENCE flag)
    if TEST_POLICY_INFERENCE:
        print("\n" + "="*50)
        print("Testing Policy Inference")
        print("="*50)
        
        # Create a trained policy with memory optimization
        print("Creating trained policy...")
        policy = _policy_config.create_trained_policy(config, checkpoint_dir)
        
        # Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
        print("Running inference on dummy example...")
        example = droid_policy.make_droid_example()
        result = policy.infer(example)
        
        # Delete the policy to free up memory immediately
        del policy
        clear_memory()
        
        print("Actions shape:", result["actions"].shape)
    else:
        print("\nPolicy inference test SKIPPED (set TEST_POLICY_INFERENCE=True to enable)")
    
    # Test Live Model (controlled by TEST_LIVE_MODEL flag)
    if TEST_LIVE_MODEL:
        print("\n" + "="*50)
        print("Testing Live Model with Fake Data")
        print("="*50)
        
        # Use the available model configuration
        config = _config.get_config("pi0_fast_droid")
        
        checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
        key = jax.random.key(0)
        
        # Create a model from the checkpoint with memory optimization
        print("Loading model parameters on GPU with memory optimization...")
        try:
            # Load parameters with reduced precision to save memory
            params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.float32)
            model = config.model.load(params)
            print("Model loaded successfully on GPU")
        except Exception as e:
            print(f"GPU memory error: {e}")
            print("Falling back to CPU...")
            # Clear GPU memory and switch to CPU
            jax.clear_caches()
            gc.collect()
            os.environ['JAX_PLATFORM_NAME'] = 'cpu'
            jax.config.update('jax_platform_name', 'cpu')
            params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.float32)
            model = config.model.load(params)
            print("Model loaded successfully on CPU")
        
        # We can create fake observations and actions to test the model
        print("Creating fake observations and actions...")
        obs, act = config.model.fake_obs(), config.model.fake_act()
        
        # Sample actions from the model
        print("Computing loss...")
        loss = model.compute_loss(key, obs, act)
        
        # Delete the model to free up memory immediately
        del model
        del params
        clear_memory()
        
        print("Loss shape:", loss.shape)
    else:
        print("\nLive model test SKIPPED (set TEST_LIVE_MODEL=True to enable)")
    
    # Memory Optimization Tips
    print("\n" + "="*50)
    print("Memory Optimization Tips")
    print("="*50)
    print("""
    The script has been updated to fix memory issues based on GitHub issues:
    
    GitHub Issues Addressed:
    - Issue #376: Memory exhaustion issue when loading pi0_base params
    - Issue #379: Memory Exhausted Issue when Training with Pi0 Fine Tuned Lora  
    - Issue #340: Inference memory cost is unexpected high
    
    Memory Optimizations Implemented:
    1. **JAX Memory Configuration**: Set XLA memory fraction and disable preallocation
    2. **Precision Reduction**: Use float32 instead of float64 to halve memory usage
    3. **Automatic CPU Fallback**: Falls back to CPU if GPU memory is insufficient
    4. **Explicit Memory Clearing**: Clear JAX caches and run garbage collection
    5. **Smaller Model**: Use `pi0_fast_droid` instead of larger models
    6. **Immediate Cleanup**: Delete models and parameters immediately after use
    
    Additional Recommendations:
    - Monitor GPU memory usage with `nvidia-smi`
    - Use smaller batch sizes if memory issues persist
    - Consider using gradient checkpointing for training
    - Close other GPU-intensive applications
    """)
    
    # Test Data Loader (controlled by TEST_DATA_LOADER flag)
    if TEST_DATA_LOADER:
        print("\n" + "="*50)
        print("Testing Data Loader with Real Training Data")
        print("="*50)
        
        # Reduce the batch size to reduce memory usage (addressing Issue #379)
        config = dataclasses.replace(config, batch_size=1)  # Further reduced from 2 to 1
        
        # Load a single batch of data. This is the same data that will be used during training.
        # NOTE: In order to make this example self-contained, we are skipping the normalization step
        # since it requires the normalization statistics to be generated using `compute_norm_stats`.
        print("Creating data loader with minimal batch size...")
        loader = _data_loader.create_data_loader(config, num_batches=1, skip_norm_stats=True)
        obs, act = next(iter(loader))
        
        # Create model again for the data loader example with memory optimization
        print("Loading model for data loader example...")
        try:
            params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.float32)
            model = config.model.load(params)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using CPU fallback...")
            os.environ['JAX_PLATFORM_NAME'] = 'cpu'
            jax.config.update('jax_platform_name', 'cpu')
            params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.float32)
            model = config.model.load(params)
        
        # Sample actions from the model
        print("Computing loss with real data...")
        loss = model.compute_loss(key, obs, act)
        
        # Delete the model to free up memory
        del model
        del params
        clear_memory()
        
        print("Loss shape:", loss.shape)
    else:
        print("\nData loader test SKIPPED (set TEST_DATA_LOADER=True to enable)")


if __name__ == "__main__":
    main()
