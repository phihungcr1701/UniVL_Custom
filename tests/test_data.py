"""Test data loading and transformations"""

import torch
import pytest
from transformers import AutoTokenizer

from data.transforms import (
    VideoNormalize,
    MaskTokens,
    MaskFrames,
    CaptionPreprocess,
    collate_fn
)


@pytest.fixture
def tokenizer():
    """Create tokenizer"""
    return AutoTokenizer.from_pretrained('bert-base-uncased')


def test_video_normalize():
    """Test video normalization"""
    transform = VideoNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    # Input shape: (num_frames, dim)
    video = torch.randn(12, 1024)
    normalized = transform(video)
    
    assert normalized.shape == video.shape
    assert not torch.allclose(normalized, video)  # Should be different


def test_mask_tokens(tokenizer):
    """Test token masking for MLM"""
    mask_transform = MaskTokens(
        tokenizer=tokenizer,
        mask_prob=0.15,
        mask_token_id=tokenizer.mask_token_id,
        vocab_size=tokenizer.vocab_size,
    )
    
    # Create sample input
    input_ids = torch.randint(100, 1000, (20,))  # Avoid special tokens
    
    masked_ids, labels = mask_transform(input_ids.clone())
    
    assert masked_ids.shape == input_ids.shape
    assert labels.shape == input_ids.shape
    
    # Check some tokens were masked
    num_masked = (labels != -100).sum().item()
    assert num_masked > 0, "No tokens were masked"
    
    # Check masked positions have mask token
    masked_positions = labels != -100
    # Note: Some might be replaced with random tokens (10% of masked)


def test_mask_frames():
    """Test frame masking for MFM"""
    mask_transform = MaskFrames(mask_prob=0.15)
    
    video = torch.randn(12, 1024)
    masked_video, labels = mask_transform(video.clone())
    
    assert masked_video.shape == video.shape
    assert labels.shape == (12,)
    
    # Check some frames were masked
    num_masked = labels.sum().item()
    assert num_masked > 0, "No frames were masked"
    
    # Check masked frames are zero
    masked_positions = labels == 1
    assert torch.allclose(masked_video[masked_positions], torch.zeros_like(masked_video[masked_positions]))


def test_caption_preprocess(tokenizer):
    """Test caption preprocessing"""
    transform = CaptionPreprocess(
        tokenizer=tokenizer,
        max_length=20,
    )
    
    caption = "A person is riding a bike"
    result = transform(caption)
    
    assert 'caption_ids' in result
    assert 'caption_mask' in result
    
    # Check shapes
    assert result['caption_ids'].shape[0] <= 20
    assert result['caption_mask'].shape[0] <= 20
    
    # Check mask is correct
    assert result['caption_mask'].sum() == len(result['caption_ids'])


def test_collate_fn(tokenizer):
    """Test batch collation"""
    # Create sample batch
    batch = [
        {
            'video_id': 'video0',
            'input_ids': torch.randint(0, 1000, (15,)),
            'attention_mask': torch.ones(15),
            'video': torch.randn(10, 1024),
            'video_mask': torch.ones(10),
            'caption_ids': torch.randint(0, 1000, (12,)),
            'caption_mask': torch.ones(12),
        },
        {
            'video_id': 'video1',
            'input_ids': torch.randint(0, 1000, (18,)),
            'attention_mask': torch.ones(18),
            'video': torch.randn(8, 1024),
            'video_mask': torch.ones(8),
            'caption_ids': torch.randint(0, 1000, (10,)),
            'caption_mask': torch.ones(10),
        },
    ]
    
    collated = collate_fn(batch)
    
    # Check all required keys
    assert 'video_ids' in collated
    assert 'input_ids' in collated
    assert 'attention_mask' in collated
    assert 'video' in collated
    assert 'video_mask' in collated
    assert 'caption_ids' in collated
    assert 'caption_mask' in collated
    
    # Check batch dimension
    batch_size = len(batch)
    assert collated['input_ids'].shape[0] == batch_size
    assert collated['video'].shape[0] == batch_size
    
    # Check padding (should pad to max in batch)
    assert collated['input_ids'].shape[1] == 18  # Max text length
    assert collated['video'].shape[1] == 10  # Max video length
    assert collated['caption_ids'].shape[1] == 12  # Max caption length


def test_collate_fn_with_pretrain(tokenizer):
    """Test batch collation with pretrain data"""
    batch = [
        {
            'video_id': 'video0',
            'input_ids': torch.randint(0, 1000, (15,)),
            'attention_mask': torch.ones(15),
            'video': torch.randn(10, 1024),
            'video_mask': torch.ones(10),
            'caption_ids': torch.randint(0, 1000, (12,)),
            'caption_mask': torch.ones(12),
            'masked_lm_labels': torch.randint(-100, 1000, (15,)),
            'masked_video': torch.randn(10, 1024),
            'video_labels_index': torch.randint(0, 2, (10,)),
        },
    ]
    
    collated = collate_fn(batch)
    
    # Check pretrain-specific keys
    assert 'masked_lm_labels' in collated
    assert 'masked_video' in collated
    assert 'video_labels_index' in collated


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
