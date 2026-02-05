"""Quick test script to verify installation and basic functionality"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import torch
from transformers import AutoTokenizer

print("=" * 60)
print("UniVL-Custom Installation Test")
print("=" * 60)

# Test 1: Check PyTorch
print("\n[1/6] Checking PyTorch...")
print(f"  ✓ PyTorch version: {torch.__version__}")
print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  ✓ CUDA version: {torch.version.cuda}")
    print(f"  ✓ GPU device: {torch.cuda.get_device_name(0)}")

# Test 2: Check imports
print("\n[2/6] Checking imports...")
try:
    from models.univl import UniVLModel
    print("  ✓ UniVLModel")
except Exception as e:
    print(f"  ✗ UniVLModel: {e}")
    sys.exit(1)

try:
    from data.msrvtt_dataset import MSRVTTCaptionDataset
    print("  ✓ MSRVTTCaptionDataset")
except Exception as e:
    print(f"  ✗ MSRVTTCaptionDataset: {e}")
    sys.exit(1)

try:
    from training.trainer import Trainer
    from training.evaluator import Evaluator
    print("  ✓ Trainer & Evaluator")
except Exception as e:
    print(f"  ✗ Trainer/Evaluator: {e}")
    sys.exit(1)

try:
    from utils.config import UniVLConfig, load_config
    print("  ✓ UniVLConfig")
except Exception as e:
    print(f"  ✗ UniVLConfig: {e}")
    sys.exit(1)

# Test 3: Load config
print("\n[3/6] Loading configuration...")
try:
    config = load_config("configs/finetune_msrvtt.yaml")
    print(f"  ✓ Loaded config from YAML")
    print(f"  ✓ Batch size: {config.training.batch_size}")
    print(f"  ✓ Learning rate: {config.training.learning_rate}")
except Exception as e:
    print(f"  ✗ Config loading failed: {e}")
    sys.exit(1)

# Test 4: Create tokenizer
print("\n[4/6] Creating tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(config.model.bert_model)
    print(f"  ✓ Tokenizer: {config.model.bert_model}")
    print(f"  ✓ Vocab size: {len(tokenizer)}")
except Exception as e:
    print(f"  ✗ Tokenizer creation failed: {e}")
    sys.exit(1)

# Test 5: Create model
print("\n[5/6] Creating model...")
try:
    model = UniVLModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model created successfully")
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,}")
except Exception as e:
    print(f"  ✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test forward pass
print("\n[6/6] Testing forward pass...")
try:
    model.eval()
    batch_size = 2
    max_words = config.data.max_words
    max_frames = config.data.max_frames
    
    # Create dummy batch
    batch = {
        'input_ids': torch.randint(0, 1000, (batch_size, max_words)),
        'attention_mask': torch.ones(batch_size, max_words),
        'video': torch.randn(batch_size, max_frames, config.model.visual_hidden_size),
        'video_mask': torch.ones(batch_size, max_frames),
        'caption_ids': torch.randint(0, 1000, (batch_size, max_words)),
        'caption_mask': torch.ones(batch_size, max_words),
    }
    
    with torch.no_grad():
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            video=batch['video'],
            video_mask=batch['video_mask'],
            caption_ids=batch['caption_ids'],
            caption_mask=batch['caption_mask'],
            is_pretrain=False,
        )
    
    print(f"  ✓ Forward pass successful")
    print(f"  ✓ Loss: {outputs['loss'].item():.4f}")
    print(f"  ✓ Output keys: {list(outputs.keys())}")
    
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test generation
print("\n[7/7] Testing caption generation...")
try:
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate_caption(
            video=batch['video'][:1],
            video_mask=batch['video_mask'][:1],
            max_length=20,
            beam_size=1,
        )
    
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"  ✓ Generation successful")
    print(f"  ✓ Generated caption: '{caption}'")
    
except Exception as e:
    print(f"  ✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("✅ All tests passed! Installation is working correctly.")
print("=" * 60)
print("\nNext steps:")
print("1. Prepare MSRVTT data in data/msrvtt/")
print("2. Run pretraining: python scripts/pretrain.py")
print("3. Run fine-tuning: python scripts/finetune.py")
print("4. Run inference: python scripts/inference.py")
print("\nFor more information, see README.md and MIGRATION.md")
