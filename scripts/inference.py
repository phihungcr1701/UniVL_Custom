"""Inference script for generating captions from videos"""

import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import UniVLModel
from src.utils import load_config, load_checkpoint


def load_video_features(video_path: str) -> torch.Tensor:
    """
    Load video features from pickle file
    
    Args:
        video_path: Path to pickle file with video features
    
    Returns:
        Video features tensor
    """
    with open(video_path, 'rb') as f:
        features = pickle.load(f)
    
    # Handle different pickle formats
    if isinstance(features, dict):
        # Try common keys
        for key in ['features', 'video', 'data']:
            if key in features:
                features = features[key]
                break
    
    # Convert to numpy if needed
    if not isinstance(features, np.ndarray):
        features = np.array(features)
    
    return torch.from_numpy(features).float()


def generate_caption(
    model,
    tokenizer,
    video_features: torch.Tensor,
    max_length: int = 20,
    beam_size: int = 5,
    device: str = "cuda",
):
    """
    Generate caption for video
    
    Args:
        model: UniVL model
        tokenizer: HuggingFace tokenizer
        video_features: Video features (frames, feature_dim)
        max_length: Maximum caption length
        beam_size: Beam size for beam search
        device: Device to run on
    
    Returns:
        Generated caption text
    """
    model.eval()
    
    # Add batch dimension and pad/truncate to max_frames
    max_frames = model.model_config.max_frames
    num_frames = video_features.size(0)
    
    # Create video tensor and mask
    video = torch.zeros(1, max_frames, video_features.size(1))
    video_mask = torch.zeros(1, max_frames, dtype=torch.long)
    
    # Fill with actual features
    actual_frames = min(num_frames, max_frames)
    video[0, :actual_frames] = video_features[:actual_frames]
    video_mask[0, :actual_frames] = 1
    
    # Move to device
    video = video.to(device)
    video_mask = video_mask.to(device)
    
    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate_caption(
            video=video,
            video_mask=video_mask,
            max_length=max_length,
            beam_size=beam_size,
            bos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.sep_token_id,
        )
    
    # Decode caption
    caption = tokenizer.decode(
        generated_ids[0].cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    
    return caption


def main():
    parser = argparse.ArgumentParser(description="Generate captions for videos")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--video_features",
        type=str,
        required=True,
        help="Path to video features pickle file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune_msrvtt.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=20,
        help="Maximum caption length",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Beam size for beam search",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)",
    )
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create model
    print("Creating model...")
    model = UniVLModel(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        device=device,
    )
    model.to(device)
    model.eval()
    
    # Load video features
    print(f"Loading video features from {args.video_features}...")
    video_features = load_video_features(args.video_features)
    print(f"Video features shape: {video_features.shape}")
    
    # Generate caption
    print("Generating caption...")
    caption = generate_caption(
        model=model,
        tokenizer=tokenizer,
        video_features=video_features,
        max_length=args.max_length,
        beam_size=args.beam_size,
        device=device,
    )
    
    # Print result
    print("\n" + "=" * 80)
    print("Generated Caption:")
    print(f"  {caption}")
    print("=" * 80)


if __name__ == "__main__":
    main()
