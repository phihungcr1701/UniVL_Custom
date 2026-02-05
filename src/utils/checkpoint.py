"""Checkpoint saving and loading utilities"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import shutil


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step: int,
    metrics: Dict[str, float],
    save_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        step: Current global step
        metrics: Dictionary of metrics (e.g., {'bleu4': 0.405})
        save_dir: Directory to save checkpoint
        filename: Checkpoint filename
        is_best: Whether this is the best model so far
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model state dict (handle DDP/DP wrapping)
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    # Create checkpoint
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }
    
    # Save checkpoint
    checkpoint_path = save_dir / filename
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save as best model if needed
    if is_best:
        best_path = save_dir / "best_model.pth"
        shutil.copyfile(checkpoint_path, best_path)
        print(f"Best model saved: {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to map tensors to
    
    Returns:
        Dictionary with epoch, step, and metrics
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state dict (handle DDP/DP wrapping)
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'metrics': checkpoint.get('metrics', {}),
    }
    
    print(f"Checkpoint loaded - Epoch: {info['epoch']}, Step: {info['step']}")
    if info['metrics']:
        print(f"Metrics: {info['metrics']}")
    
    return info


def load_pretrained_weights(
    checkpoint_path: str,
    model: torch.nn.Module,
    strict: bool = False,
    device: str = "cpu",
):
    """
    Load pretrained weights (for initializing from pretrained model)
    
    Args:
        checkpoint_path: Path to checkpoint file or state dict
        model: PyTorch model to load weights into
        strict: Whether to strictly enforce key matching
        device: Device to map tensors to
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Pretrained weights not found: {checkpoint_path}")
    
    print(f"Loading pretrained weights: {checkpoint_path}")
    
    # Try to load as checkpoint first, fallback to state dict
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    except:
        state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict
    if hasattr(model, 'module'):
        missing, unexpected = model.module.load_state_dict(state_dict, strict=strict)
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    
    print("Pretrained weights loaded successfully")
