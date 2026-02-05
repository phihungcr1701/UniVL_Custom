"""
Script to download and prepare MSRVTT dataset for UniVL training

This script will:
1. Download MSRVTT annotations (captions)
2. Guide you to download video features (S3D)
3. Organize data in the expected structure
"""

import os
import json
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm


def download_file(url, dest_path, desc="Downloading"):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_msrvtt_annotations(data_dir):
    """Download MSRVTT annotation files"""
    print("\n[1/3] Downloading MSRVTT annotations...")
    
    # URLs for MSRVTT annotations
    urls = {
        'train_val': 'https://github.com/ArrowLuo/CLIP4Clip/raw/master/data/MSRVTT/msrvtt_data/MSRVTT_train.9k.csv',
        'test': 'https://github.com/ArrowLuo/CLIP4Clip/raw/master/data/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv',
        'caption': 'https://github.com/ArrowLuo/CLIP4Clip/raw/master/data/MSRVTT/msrvtt_data/MSRVTT_data.json',
    }
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for name, url in urls.items():
        dest = data_dir / url.split('/')[-1]
        if dest.exists():
            print(f"  ✓ {dest.name} already exists, skipping")
        else:
            try:
                download_file(url, dest, desc=f"  Downloading {dest.name}")
                print(f"  ✓ Downloaded {dest.name}")
            except Exception as e:
                print(f"  ✗ Failed to download {dest.name}: {e}")
                print(f"    Please manually download from: {url}")
    
    print("  ✓ Annotations downloaded")


def create_directory_structure(data_dir):
    """Create expected directory structure"""
    print("\n[2/3] Creating directory structure...")
    
    data_dir = Path(data_dir)
    
    # Create directories
    dirs = [
        data_dir / "annotations",
        data_dir / "videos",
        data_dir / "features",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created {d}")


def print_next_steps(data_dir):
    """Print instructions for manual steps"""
    print("\n[3/3] Next steps (manual):")
    print("\n" + "="*70)
    
    print("\n📹 DOWNLOAD VIDEO FEATURES:")
    print("  UniVL uses pre-extracted S3D features from videos.")
    print("\n  Option 1: Download pre-extracted features")
    print("    - Original repo: https://github.com/microsoft/UniVL")
    print("    - Look for MSRVTT S3D features (*.pkl files)")
    print(f"    - Place in: {Path(data_dir).absolute() / 'features'}")
    print("\n  Option 2: Extract features yourself")
    print("    - Download MSRVTT videos from: http://ms-multimedia-challenge.com/2017/dataset")
    print("    - Use S3D model to extract features")
    print("    - Save as pickle files: video{id}.pkl")
    print("\n  Feature format expected:")
    print("    - File: video0.pkl, video1.pkl, ..., video9999.pkl")
    print("    - Shape: (num_frames, 1024) where num_frames ≤ 48")
    print("    - Type: numpy array or torch tensor")
    
    print("\n📁 EXPECTED DIRECTORY STRUCTURE:")
    print(f"""
  {data_dir}/
    ├── MSRVTT_train.9k.csv          (train/val split)
    ├── MSRVTT_JSFUSION_test.csv     (test split)
    ├── MSRVTT_data.json             (all captions)
    └── features/
        ├── video0.pkl
        ├── video1.pkl
        ├── ...
        └── video9999.pkl
    """)
    
    print("\n✅ VERIFY SETUP:")
    print("  Run this command to check if data is ready:")
    print(f"    python -c \"from src.data import create_dataloader; print('Data OK!')\"")
    
    print("\n🚀 START TRAINING:")
    print("  Once data is ready, run:")
    print("    python scripts/pretrain.py --config configs/pretrain_msrvtt.yaml")
    print("    python scripts/finetune.py --config configs/finetune_msrvtt.yaml")
    
    print("\n" + "="*70)


def main():
    """Main function"""
    print("="*70)
    print("MSRVTT Data Preparation for UniVL-Custom")
    print("="*70)
    
    # Get data directory
    default_dir = "data/msrvtt"
    data_dir = input(f"\nEnter data directory (default: {default_dir}): ").strip()
    if not data_dir:
        data_dir = default_dir
    
    # Download annotations
    try:
        download_msrvtt_annotations(data_dir)
    except Exception as e:
        print(f"\n⚠ Warning: Annotation download failed: {e}")
        print("  You may need to download manually")
    
    # Create directory structure
    create_directory_structure(data_dir)
    
    # Print next steps
    print_next_steps(data_dir)
    
    print("\n✅ Setup script completed!")


if __name__ == '__main__':
    main()
