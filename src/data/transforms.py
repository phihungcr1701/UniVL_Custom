"""Data transformations for video and text"""

import torch
import torch.nn.functional as F
import random
from typing import Dict, Tuple


class VideoNormalize:
    """Normalize video features with L2 norm"""
    
    def __init__(self):
        pass
    
    def __call__(self, video: torch.Tensor, max_frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video: Video features (num_frames, feature_dim)
            max_frames: Maximum number of frames
        
        Returns:
            video: Normalized and padded video (max_frames, feature_dim)
            video_mask: Mask indicating valid frames (max_frames,)
        """
        # Convert to tensor if numpy
        if not isinstance(video, torch.Tensor):
            video = torch.from_numpy(video)
        
        video = video.float()
        
        # L2 normalization
        video = F.normalize(video, p=2, dim=-1)
        
        # Get actual length
        actual_len = min(len(video), max_frames)
        
        # Truncate or pad
        if len(video) > max_frames:
            video = video[:max_frames]
        elif len(video) < max_frames:
            padding = torch.zeros(max_frames - len(video), video.shape[-1])
            video = torch.cat([video, padding], dim=0)
        
        # Create mask
        video_mask = torch.zeros(max_frames)
        video_mask[:actual_len] = 1
        
        return video, video_mask


class MaskTokens:
    """Masked Language Modeling (MLM) transformation"""
    
    def __init__(
        self,
        tokenizer,
        mask_prob: float = 0.15,
        mask_token_id: int = None,
        vocab_size: int = None,
    ):
        """
        Args:
            tokenizer: BERT tokenizer
            mask_prob: Probability of masking a token
            mask_token_id: ID of [MASK] token
            vocab_size: Size of vocabulary
        """
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id or tokenizer.vocab['[MASK]']
        self.vocab_size = vocab_size or len(tokenizer.vocab)
        self.cls_token_id = tokenizer.vocab['[CLS]']
        self.sep_token_id = tokenizer.vocab['[SEP]']
        self.pad_token_id = 0
    
    def __call__(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply MLM masking
        
        Args:
            input_ids: Token IDs (seq_len,)
        
        Returns:
            Dict with masked_input_ids and labels
        """
        labels = input_ids.clone()
        masked_input_ids = input_ids.clone()
        
        # Create probability matrix
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        
        # Don't mask special tokens
        special_tokens_mask = (
            (input_ids == self.cls_token_id) |
            (input_ids == self.sep_token_id) |
            (input_ids == self.pad_token_id)
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Sample tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set labels to -1 for non-masked tokens (will be ignored in loss)
        labels[~masked_indices] = -1
        
        # 80% of the time: replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        masked_input_ids[indices_replaced] = self.mask_token_id
        
        # 10% of the time: replace with random token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool() &
            masked_indices &
            ~indices_replaced
        )
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        masked_input_ids[indices_random] = random_words[indices_random]
        
        # 10% of the time: keep original token (do nothing)
        
        return {
            'masked_input_ids': masked_input_ids,
            'labels': labels
        }


class MaskFrames:
    """Masked Frame Modeling (MFM) transformation"""
    
    def __init__(self, mask_prob: float = 0.15):
        """
        Args:
            mask_prob: Probability of masking a frame
        """
        self.mask_prob = mask_prob
    
    def __call__(self, video: torch.Tensor, video_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply MFM masking
        
        Args:
            video: Video features (max_frames, feature_dim)
            video_mask: Valid frame mask (max_frames,)
        
        Returns:
            Dict with masked_video and labels
        """
        masked_video = video.clone()
        labels = torch.full((video.shape[0],), -1, dtype=torch.long)
        
        # Only mask valid frames
        valid_frames = video_mask.bool()
        num_valid = valid_frames.sum().item()
        
        if num_valid == 0:
            return {
                'masked_video': masked_video,
                'labels': labels
            }
        
        # Sample frames to mask
        mask_prob = torch.full((video.shape[0],), self.mask_prob)
        mask_prob[~valid_frames] = 0.0  # Don't mask padding frames
        
        masked_indices = torch.bernoulli(mask_prob).bool()
        
        # Zero out masked frames
        masked_video[masked_indices] = 0.0
        
        # Set labels (frame index for masked frames, -1 for others)
        labels[masked_indices] = torch.arange(video.shape[0])[masked_indices]
        
        return {
            'masked_video': masked_video,
            'labels': labels
        }


class CaptionPreprocess:
    """Preprocess captions for decoder"""
    
    def __init__(self, tokenizer, max_length: int = 20):
        """
        Args:
            tokenizer: BERT tokenizer
            max_length: Maximum caption length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, caption: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize and prepare caption for decoder
        
        Args:
            caption: Caption text
        
        Returns:
            Dict with decoder_input_ids, decoder_target_ids, decoder_mask
        """
        # Tokenize
        caption_tokens = self.tokenizer.tokenize(caption)
        
        # Truncate if needed
        max_caption_len = self.max_length - 1  # Reserve space for [CLS]/[SEP]
        if len(caption_tokens) > max_caption_len:
            caption_tokens = caption_tokens[:max_caption_len]
        
        # Decoder input: [CLS] + tokens
        input_tokens = ['[CLS]'] + caption_tokens
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        
        # Decoder target: tokens + [SEP]
        output_tokens = caption_tokens + ['[SEP]']
        output_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)
        
        # Decoder mask
        decoder_mask = [1] * len(input_ids)
        
        # Padding
        while len(input_ids) < self.max_length:
            input_ids.append(0)
            output_ids.append(0)
            decoder_mask.append(0)
        
        return {
            'decoder_input_ids': torch.tensor(input_ids, dtype=torch.long),
            'decoder_target_ids': torch.tensor(output_ids, dtype=torch.long),
            'decoder_mask': torch.tensor(decoder_mask, dtype=torch.long)
        }


def collate_fn(batch):
    """
    Collate function for DataLoader
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched dict
    """
    # Get keys from first sample
    keys = batch[0].keys()
    
    # Initialize batch dict
    batched = {}
    
    for key in keys:
        if key == 'video_id':
            # Keep as list
            batched[key] = [sample[key] for sample in batch]
        else:
            # Stack tensors
            batched[key] = torch.stack([sample[key] for sample in batch])
    
    return batched
