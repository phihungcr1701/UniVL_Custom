"""Basic model tests"""

import torch
import pytest
from transformers import AutoTokenizer

from models.univl import UniVLModel
from utils.config import UniVLConfig


@pytest.fixture
def config():
    """Create a minimal test config"""
    return UniVLConfig(
        model={
            'bert_model': 'bert-base-uncased',
            'visual_hidden_size': 1024,
            'visual_num_hidden_layers': 2,
            'cross_num_hidden_layers': 2,
            'decoder_num_hidden_layers': 2,
        },
        data={
            'data_dir': 'data/msrvtt',
            'max_words': 20,
            'max_frames': 12,
        },
        training={
            'batch_size': 2,
        }
    )


@pytest.fixture
def tokenizer():
    """Create tokenizer"""
    return AutoTokenizer.from_pretrained('bert-base-uncased')


@pytest.fixture
def model(config):
    """Create model"""
    return UniVLModel(config)


@pytest.fixture
def sample_batch(tokenizer):
    """Create a sample batch for testing"""
    batch_size = 2
    max_words = 20
    max_frames = 12
    visual_dim = 1024
    
    return {
        'input_ids': torch.randint(0, 30000, (batch_size, max_words)),
        'attention_mask': torch.ones(batch_size, max_words),
        'video': torch.randn(batch_size, max_frames, visual_dim),
        'video_mask': torch.ones(batch_size, max_frames),
        'caption_ids': torch.randint(0, 30000, (batch_size, max_words)),
        'caption_mask': torch.ones(batch_size, max_words),
    }


def test_model_creation(model):
    """Test that model can be created"""
    assert model is not None
    assert hasattr(model, 'text_encoder')
    assert hasattr(model, 'visual_encoder')
    assert hasattr(model, 'cross_encoder')
    assert hasattr(model, 'decoder')


def test_model_forward_caption(model, sample_batch):
    """Test forward pass for caption task"""
    outputs = model(
        input_ids=sample_batch['input_ids'],
        attention_mask=sample_batch['attention_mask'],
        video=sample_batch['video'],
        video_mask=sample_batch['video_mask'],
        caption_ids=sample_batch['caption_ids'],
        caption_mask=sample_batch['caption_mask'],
        is_pretrain=False,
    )
    
    assert 'loss' in outputs
    assert 'caption_loss' in outputs
    assert outputs['loss'].dim() == 0  # Scalar loss
    
    # Check loss is reasonable
    assert outputs['loss'].item() > 0
    assert not torch.isnan(outputs['loss'])


def test_model_forward_pretrain(model, sample_batch):
    """Test forward pass for pretrain task"""
    # Add masked tokens for pretraining
    sample_batch['masked_lm_labels'] = torch.randint(0, 30000, (2, 20))
    sample_batch['masked_video'] = sample_batch['video'].clone()
    sample_batch['video_labels_index'] = torch.randint(0, 2, (2, 12))
    
    outputs = model(
        input_ids=sample_batch['input_ids'],
        attention_mask=sample_batch['attention_mask'],
        video=sample_batch['video'],
        video_mask=sample_batch['video_mask'],
        caption_ids=sample_batch['caption_ids'],
        caption_mask=sample_batch['caption_mask'],
        is_pretrain=True,
        masked_lm_labels=sample_batch['masked_lm_labels'],
        masked_video=sample_batch['masked_video'],
        video_labels_index=sample_batch['video_labels_index'],
    )
    
    assert 'loss' in outputs
    assert 'caption_loss' in outputs
    assert 'mlm_loss' in outputs
    assert 'mfm_loss' in outputs
    
    # Check all losses are reasonable
    for loss_name in ['loss', 'caption_loss', 'mlm_loss', 'mfm_loss']:
        assert outputs[loss_name].dim() == 0
        assert outputs[loss_name].item() > 0
        assert not torch.isnan(outputs[loss_name])


def test_caption_generation(model, sample_batch):
    """Test caption generation"""
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate_caption(
            video=sample_batch['video'][:1],  # Single sample
            video_mask=sample_batch['video_mask'][:1],
            max_length=10,
            beam_size=1,  # Greedy for speed
        )
    
    assert generated_ids.shape[0] == 1  # Batch size
    assert generated_ids.shape[1] <= 10  # Max length
    assert generated_ids.dtype == torch.long


def test_beam_search_generation(model, sample_batch, tokenizer):
    """Test beam search generation"""
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate_caption(
            video=sample_batch['video'][:1],
            video_mask=sample_batch['video_mask'][:1],
            max_length=15,
            beam_size=3,
        )
    
    # Decode to text
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    assert isinstance(caption, str)
    assert len(caption) > 0


def test_model_save_load(model, tmp_path):
    """Test model can be saved and loaded"""
    save_path = tmp_path / "test_model.pth"
    
    # Save
    torch.save(model.state_dict(), save_path)
    
    # Load
    loaded_model = UniVLModel(model.config)
    loaded_model.load_state_dict(torch.load(save_path))
    
    # Compare parameters
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded_model.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1, p2)


def test_model_device_transfer(model):
    """Test model can be moved to different devices"""
    # CPU
    model_cpu = model.cpu()
    assert next(model_cpu.parameters()).device.type == 'cpu'
    
    # CUDA (if available)
    if torch.cuda.is_available():
        model_cuda = model.cuda()
        assert next(model_cuda.parameters()).device.type == 'cuda'


def test_gradient_flow(model, sample_batch):
    """Test gradients flow through the model"""
    model.train()
    
    outputs = model(
        input_ids=sample_batch['input_ids'],
        attention_mask=sample_batch['attention_mask'],
        video=sample_batch['video'],
        video_mask=sample_batch['video_mask'],
        caption_ids=sample_batch['caption_ids'],
        caption_mask=sample_batch['caption_mask'],
        is_pretrain=False,
    )
    
    # Backward
    loss = outputs['loss']
    loss.backward()
    
    # Check gradients exist
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            has_grad = True
    
    assert has_grad, "No parameters have gradients"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
