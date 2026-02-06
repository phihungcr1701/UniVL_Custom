# UniVL Video Captioning - Custom Implementation

Modern, clean implementation of UniVL for MSRVTT video captioning with full pretrain and fine-tuning pipeline.

## 🚀 Features

- ✅ **Modern Stack**: PyTorch 2.x + HuggingFace Transformers
- ✅ **YAML Configuration**: Reproducible experiment configs
- ✅ **Logging**: W&B and TensorBoard integration
- ✅ **Full Pipeline**: Pretrain (MLM/MFM) + Fine-tuning (Caption)
- ✅ **Clean Architecture**: Modular, testable, maintainable
- ✅ **MSRVTT Dataset**: Optimized for MSRVTT video captioning

## 📦 Installation

```bash
cd UniVL-custom
pip install -r requirements.txt
```

## 🏃 Quick Start

### 1. Prepare Data

**IMPORTANT**: MSRVTT uses a single data file with fixed video_id splits:
- **Train**: video0 - video6512 (6513 videos)
- **Val**: video6513 - video7009 (497 videos)
- **Test**: video7010 - video9999 (2990 videos)

```bash
# Download MSRVTT data
python scripts/download_data.py

# Expected structure:
# data/msrvtt/
# ├── MSRVTT_data.json          # Single file with all videos/captions
# └── features/                 # S3D video features
#     ├── video0.pkl
#     ├── video1.pkl
#     └── ... (video9999.pkl)
```

### 2. Pretrain on MSRVTT

```bash
python scripts/pretrain.py --config configs/pretrain_msrvtt.yaml
```

### 3. Fine-tune for Captioning

```bash
python scripts/finetune.py --config configs/finetune_msrvtt.yaml \
    --init_model checkpoints/pretrain/best_model.pth
```

### 4. Inference

```bash
python scripts/inference.py \
    --checkpoint checkpoints/finetune/best_model.pth \
    --video_features path/to/video_features.pkl
```

## 📁 Project Structure

```
UniVL-custom/
├── configs/               # YAML configuration files
│   ├── base.yaml         # Base configuration
│   ├── pretrain_msrvtt.yaml
│   └── finetune_msrvtt.yaml
├── src/
│   ├── models/           # Model architecture
│   │   ├── encoders.py   # Text/Visual/Cross encoders
│   │   ├── decoder.py    # Caption decoder
│   │   └── univl.py      # Main UniVL model
│   ├── data/             # Dataset and data loading
│   │   ├── transforms.py # Data preprocessing
│   │   └── msrvtt_dataset.py
│   ├── training/         # Training logic
│   │   ├── trainer.py
│   │   └── evaluator.py
│   └── utils/            # Utilities
│       ├── config.py     # Config management
│       ├── logger.py     # Logging setup
│       └── checkpoint.py # Model checkpointing
├── scripts/              # Entry points
│   ├── pretrain.py
│   ├── finetune.py
│   └── inference.py
└── tests/                # Unit tests
```

## ⚙️ Configuration

All hyperparameters are managed via YAML files in `configs/`. Key settings:

- **Model**: Number of layers, hidden sizes, dropout
- **Training**: Learning rate, batch size, epochs, warmup
- **Data**: Paths to CSV/JSON/features, max_words, max_frames
- **Logging**: W&B project name, TensorBoard directory

Example:
```yaml
model:
  text_num_hidden_layers: 12
  visual_num_hidden_layers: 6
  cross_num_hidden_layers: 2
  decoder_num_hidden_layers: 3
  hidden_size: 768

training:
  learning_rate: 3.0e-5
  batch_size: 128
  epochs: 5

data:
  data_dir: "data/msrvtt"
  data_path: "MSRVTT_data.json"  # Single file, split by video_id range
  max_words: 48
  max_frames: 48
```
  batch_size: 128
  learning_rate: 3e-5
  epochs: 5
  warmup_ratio: 0.1
```

## 📊 Results (Expected)

| Model | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|-------|--------|--------|---------|-------|
| UniVL (original) | 40.5 | 28.8 | 60.9 | 47.1 |
| UniVL (custom) | TBD | TBD | TBD | TBD |

## 🔄 Migration from Original Code

Key differences from original UniVL implementation:
- ✅ Removed retrieval task code (similarity losses, cross-similarity)
- ✅ Replaced custom BERT with HuggingFace transformers
- ✅ YAML configs instead of argparse
- ✅ Modern PyTorch patterns (AMP, DDP)
- ✅ Improved logging and checkpointing

## 📝 License

Apache 2.0 (same as original UniVL)

## 🙏 Acknowledgements

Based on the original [UniVL](https://github.com/microsoft/UniVL) implementation by Microsoft Research.
