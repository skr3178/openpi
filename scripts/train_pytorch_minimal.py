"""
Minimal training script for testing with low memory requirements.

This script uses the PI0PytorchMinimal model which replaces heavy components
with tiny transformers, allowing you to test training logic without needing
a high-end GPU.

Usage:
    # CPU training (very slow but works everywhere)
    python scripts/train_pytorch_minimal.py --device cpu --num_steps 100
    
    # GPU training (fast and low memory ~100MB)
    python scripts/train_pytorch_minimal.py --device cuda --num_steps 1000
    
    # With custom config
    python scripts/train_pytorch_minimal.py --batch_size 8 --lr 3e-4 --num_steps 2000
"""

import argparse
import logging
import pathlib
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from openpi.models_pytorch.pi0_pytorch_minimal import (
    PI0PytorchMinimal,
    MinimalConfig,
    SimpleObservation,
)


def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def create_fake_batch(config: MinimalConfig, batch_size: int, device: str):
    """Create a fake batch of training data."""
    # Create fake images
    images = {
        "base_0_rgb": torch.randn(batch_size, 224, 224, 3, device=device),
        "left_wrist_0_rgb": torch.randn(batch_size, 224, 224, 3, device=device),
        "right_wrist_0_rgb": torch.randn(batch_size, 224, 224, 3, device=device),
    }
    
    image_masks = {
        key: torch.ones(batch_size, dtype=torch.bool, device=device) 
        for key in images.keys()
    }
    
    state = torch.randn(batch_size, config.action_dim, device=device)
    
    tokenized_prompt = torch.randint(
        0, 2048, (batch_size, config.max_token_len), dtype=torch.int32, device=device
    )
    
    tokenized_prompt_mask = torch.ones(
        batch_size, config.max_token_len, dtype=torch.bool, device=device
    )
    
    observation = SimpleObservation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )
    
    actions = torch.randn(
        batch_size, config.action_horizon, config.action_dim, 
        dtype=torch.float32, device=device
    )
    
    return observation, actions


def train(args):
    """Main training function."""
    setup_logging()
    
    # Create config
    config = MinimalConfig(
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        max_token_len=args.max_token_len,
        hidden_dim=args.hidden_dim,
    )
    
    logging.info("="*70)
    logging.info("MINIMAL PI0 TRAINING")
    logging.info("="*70)
    logging.info(f"Device: {args.device}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Learning rate: {args.lr}")
    logging.info(f"Training steps: {args.num_steps}")
    logging.info(f"Model config: action_dim={config.action_dim}, "
                 f"action_horizon={config.action_horizon}, "
                 f"hidden_dim={config.hidden_dim}")
    
    # Create model
    logging.info("Creating model...")
    model = PI0PytorchMinimal(config).to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,} (~{total_params * 4 / 1024 / 1024:.1f} MB)")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler (cosine decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps, eta_min=args.lr * 0.1
    )
    
    # Create checkpoint directory
    if args.save_dir:
        save_path = pathlib.Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Checkpoints will be saved to: {save_path}")
    
    # Training loop
    logging.info("Starting training...")
    model.train()
    
    start_time = time.time()
    losses = []
    
    pbar = tqdm(range(args.num_steps), desc="Training")
    
    for step in pbar:
        # Create batch
        observation, actions = create_fake_batch(config, args.batch_size, args.device)
        
        # Forward pass
        loss_tensor = model(observation, actions)
        loss = loss_tensor.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        
        # Log metrics
        losses.append(loss.item())
        
        if (step + 1) % args.log_interval == 0:
            avg_loss = sum(losses) / len(losses)
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            current_lr = scheduler.get_last_lr()[0]
            
            logging.info(
                f"Step {step + 1}/{args.num_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Speed: {steps_per_sec:.2f} steps/s"
            )
            
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.2e}",
            })
            
            losses = []
        
        # Save checkpoint
        if args.save_dir and (step + 1) % args.save_interval == 0:
            checkpoint = {
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": vars(args),
            }
            
            checkpoint_path = save_path / f"checkpoint_{step + 1}.pt"
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")
    
    pbar.close()
    
    # Final evaluation
    logging.info("="*70)
    logging.info("Training complete! Running final evaluation...")
    
    model.eval()
    with torch.no_grad():
        observation, actions = create_fake_batch(config, args.batch_size, args.device)
        
        # Forward pass
        loss_tensor = model(observation, actions)
        final_loss = loss_tensor.mean().item()
        
        # Sample actions
        sampled_actions = model.sample_actions(
            device=args.device,
            observation=observation,
            num_steps=10
        )
        
        logging.info(f"Final loss: {final_loss:.4f}")
        logging.info(f"Sampled actions shape: {sampled_actions.shape}")
        logging.info(f"Sampled actions range: [{sampled_actions.min():.2f}, {sampled_actions.max():.2f}]")
    
    # Save final model
    if args.save_dir:
        final_path = save_path / "final_model.pt"
        torch.save(model.state_dict(), final_path)
        logging.info(f"Saved final model to {final_path}")
    
    total_time = time.time() - start_time
    logging.info(f"Total training time: {total_time:.1f}s ({total_time / args.num_steps:.3f}s per step)")
    logging.info("="*70)
    
    # Memory stats (if using GPU)
    if torch.cuda.is_available() and args.device == "cuda":
        logging.info("GPU Memory Usage:")
        logging.info(f"  Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        logging.info(f"  Current memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        logging.info("="*70)


def main():
    parser = argparse.ArgumentParser(description="Train minimal PI0 model")
    
    # Device config
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to train on (cuda/cpu)")
    
    # Model config
    parser.add_argument("--action_dim", type=int, default=7,
                       help="Action dimension")
    parser.add_argument("--action_horizon", type=int, default=10,
                       help="Action horizon")
    parser.add_argument("--max_token_len", type=int, default=16,
                       help="Maximum token length")
    parser.add_argument("--hidden_dim", type=int, default=128,
                       help="Hidden dimension")
    
    # Training config
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--num_steps", type=int, default=1000,
                       help="Number of training steps")
    
    # Logging config
    parser.add_argument("--log_interval", type=int, default=50,
                       help="How often to log metrics")
    parser.add_argument("--save_interval", type=int, default=500,
                       help="How often to save checkpoints")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Directory to save checkpoints (optional)")
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()


