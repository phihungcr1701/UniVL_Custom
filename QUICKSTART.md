# Quick Start Guide

## Installation

```bash
# Clone the repository (if needed)
cd UniVL-custom

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_installation.py
```

## Data Preparation

### Option 1: Automated (Recommended)
```bash
python scripts/download_data.py
```

### Option 2: Manual
1. Download MSRVTT annotations:
   - [MSRVTT_train.9k.csv](https://github.com/ArrowLuo/CLIP4Clip/raw/master/data/MSRVTT/msrvtt_data/MSRVTT_train.9k.csv)
   - [MSRVTT_JSFUSION_test.csv](https://github.com/ArrowLuo/CLIP4Clip/raw/master/data/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv)
   - [MSRVTT_data.json](https://github.com/ArrowLuo/CLIP4Clip/raw/master/data/MSRVTT/msrvtt_data/MSRVTT_data.json)

2. Download S3D features (contact original authors or extract yourself)
   - Place in `data/msrvtt/features/video{id}.pkl`

## Training

### Pretraining (Optional but recommended)
```bash
python scripts/pretrain.py --config configs/pretrain_msrvtt.yaml
```

Monitor with TensorBoard:
```bash
tensorboard --logdir runs/
```

Or W&B (if enabled):
```bash
wandb login
# Training will automatically log to W&B
```

### Fine-tuning
```bash
# From scratch
python scripts/finetune.py --config configs/finetune_msrvtt.yaml

# From pretrained checkpoint
python scripts/finetune.py \
    --config configs/finetune_msrvtt.yaml \
    --init_model checkpoints/pretrain/best_model.pth
```

## Inference

### Single video
```bash
python scripts/inference.py \
    --checkpoint checkpoints/finetune/best_model.pth \
    --video_features data/msrvtt/features/video0.pkl \
    --beam_size 5
```

### Batch inference
```python
from src.models import UniVLModel
from src.utils import load_config, load_checkpoint
import torch

# Load model
config = load_config("configs/finetune_msrvtt.yaml")
model = UniVLModel(config)
load_checkpoint("checkpoints/finetune/best_model.pth", model)
model.eval()

# Generate caption
video = torch.load("data/msrvtt/features/video0.pkl")  # (T, 1024)
video = video.unsqueeze(0)  # (1, T, 1024)
video_mask = torch.ones(1, video.size(1))

with torch.no_grad():
    caption_ids = model.generate_caption(
        video=video,
        video_mask=video_mask,
        max_length=20,
        beam_size=5,
    )

caption = tokenizer.decode(caption_ids[0], skip_special_tokens=True)
print(f"Caption: {caption}")
```

## Configuration

All hyperparameters are in YAML files:
- `configs/base.yaml` - Base configuration
- `configs/pretrain_msrvtt.yaml` - Pretraining settings
- `configs/finetune_msrvtt.yaml` - Fine-tuning settings

Override in command line:
```bash
python scripts/finetune.py \
    --config configs/finetune_msrvtt.yaml \
    training.learning_rate=1e-5 \
    training.batch_size=64
```

## Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_model.py::test_model_forward_caption -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Common Issues

### Issue: CUDA out of memory
**Solution**: Reduce batch size or enable gradient accumulation
```yaml
training:
  batch_size: 32  # Reduce from 128
  gradient_accumulation_steps: 4  # Effective batch = 32 * 4 = 128
```

### Issue: Import errors
**Solution**: Make sure you're in the project root and Python path is set
```bash
cd UniVL-custom
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Issue: W&B login required
**Solution**: Either login or disable W&B
```bash
# Option 1: Login
wandb login

# Option 2: Disable
export WANDB_MODE=disabled
```

### Issue: Tokenizer downloads slowly
**Solution**: Set HuggingFace cache directory
```bash
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache
```

## Performance Tips

1. **Enable mixed precision** (2x speedup):
   ```yaml
   training:
     use_amp: true
   ```

2. **Use gradient accumulation** (same memory, larger effective batch):
   ```yaml
   training:
     batch_size: 32
     gradient_accumulation_steps: 4  # Effective = 128
   ```

3. **Distributed training** (multi-GPU):
   ```bash
   python -m torch.distributed.launch \
       --nproc_per_node=4 \
       scripts/finetune.py --config configs/finetune_msrvtt.yaml
   ```

4. **Reduce sequence lengths** (faster but may hurt performance):
   ```yaml
   data:
     max_words: 32  # Reduce from 48
     max_frames: 32  # Reduce from 48
   ```

## Expected Results

On MSRVTT test set (after pretraining + fine-tuning):
- BLEU-4: ~0.40
- METEOR: ~0.28
- ROUGE-L: ~0.60
- CIDEr: ~0.47

Training time (single V100):
- Pretraining: ~6 hours (5 epochs)
- Fine-tuning: ~2 hours (5 epochs)

## Directory Structure

```
UniVL-custom/
├── configs/              # YAML configuration files
├── src/
│   ├── models/          # Model architecture
│   ├── data/            # Data loading and transforms
│   ├── training/        # Trainer and Evaluator
│   └── utils/           # Config, logger, checkpoint utils
├── scripts/             # Entry points (pretrain, finetune, inference)
├── tests/               # Unit tests
├── data/                # Dataset directory (not in repo)
├── checkpoints/         # Saved models (not in repo)
├── runs/                # TensorBoard logs (not in repo)
└── wandb/               # W&B logs (not in repo)
```

## Citation

If you use this code, please cite the original UniVL paper:
```
@inproceedings{luo2020univl,
  title={UniVL: A Unified Video and Language Pre-Training Model for Multimodal Understanding and Generation},
  author={Luo, Huaishao and Ji, Lei and Shi, Botian and Huang, Hao and Duan, Nan and Li, Tengchao and Chen, Xilin and Zhou, Ming},
  booktitle={arXiv preprint arXiv:2002.06353},
  year={2020}
}
```

## License

Same as original UniVL - MIT License

## Support

- Check [README.md](README.md) for project overview
- Check [MIGRATION.md](MIGRATION.md) for migrating from original code
- Open an issue for bugs or questions
