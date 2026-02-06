"""MSRVTT Dataset for video captioning"""

import os
import json
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from transformers import BertTokenizer

from .transforms import VideoNormalize, MaskTokens, MaskFrames, CaptionPreprocess, collate_fn


class MSRVTTCaptionDataset(Dataset):
    """
    MSRVTT dataset for video captioning
    
    IMPORTANT: Unlike typical datasets, MSRVTT uses a SINGLE data file
    but splits by video_id range:
    - Train: video0 - video6512 (6513 videos)
    - Val: video6513 - video7009 (497 videos)
    - Test: video7010 - video9999 (2990 videos)
    
    This matches the original UniVL implementation in main_task_caption_test.py
    """
    
    def __init__(
        self,
        data_path: str,
        features_path: str,
        tokenizer: BertTokenizer,
        max_words: int = 20,
        max_frames: int = 100,
        split_type: str = "train",
        is_pretrain: bool = False,
    ):
        """
        Args:
            data_path: Path to MSRVTT_data.json (contains all videos and captions)
            features_path: Path to video features (.pkl file with all videos)
            tokenizer: BERT tokenizer
            max_words: Maximum number of words per caption
            max_frames: Maximum number of video frames
            split_type: "train", "val", or "test"
            is_pretrain: Whether this is for pretraining (affects data augmentation)
        """
        assert split_type in ["train", "val", "test"], f"Invalid split_type: {split_type}"
        
        self.data_path = data_path
        self.features_path = features_path
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_frames = max_frames
        self.split_type = split_type
        self.is_pretrain = is_pretrain
        
        # Load data
        self.data = json.load(open(data_path, 'r'))
        self.feature_dict = pickle.load(open(features_path, 'rb'))
        
        # Define video_id splits (FIXED ranges, same as original)
        # Train: video0 : video6512 (6513)
        # Val: video6513 : video7009 (497)
        # Test: video7010 : video9999 (2990)
        video_ids = [self.data['videos'][idx]['video_id'] for idx in range(len(self.data['videos']))]
        
        split_dict = {
            "train": video_ids[:6513],
            "val": video_ids[6513:6513 + 497],
            "test": video_ids[6513 + 497:]
        }
        
        self.video_ids = split_dict[split_type]
        
        # Build samples
        self.samples = []
        self.video_captions = defaultdict(list)
        
        # Collect all captions for videos in this split
        for sent_item in self.data['sentences']:
            if sent_item['video_id'] in self.video_ids:
                self.video_captions[sent_item['video_id']].append(sent_item['caption'])
        
        # Build sample list
        if split_type == "train":
            # Training: Use ALL captions (expand to multiple samples per video)
            for sent_item in self.data['sentences']:
                if sent_item['video_id'] in self.video_ids:
                    self.samples.append({
                        'video_id': sent_item['video_id'],
                        'caption': sent_item['caption']
                    })
        else:
            # Val/Test: Use only FIRST caption per video
            for vid in self.video_ids:
                if vid in self.video_captions and len(self.video_captions[vid]) > 0:
                    self.samples.append({
                        'video_id': vid,
                        'caption': self.video_captions[vid][0]
                    })
        
        # Initialize transforms
        self.video_transform = VideoNormalize()
        self.caption_transform = CaptionPreprocess(tokenizer, max_length=max_words)
        
        if is_pretrain:
            self.mask_tokens = MaskTokens(tokenizer, mask_prob=0.15)
            self.mask_frames = MaskFrames(mask_prob=0.15)
        
        print(f"[MSRVTT {split_type.upper()}] Loaded {len(self.samples)} samples from {len(self.video_ids)} videos")
    
    def __len__(self):
        return len(self.samples)
    
    def _get_text_features(self, caption: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize caption for encoder input
        
        Returns dict with:
            - input_ids: Token IDs with [CLS] + caption + [SEP]
            - attention_mask: Mask for valid tokens
            - token_type_ids: Segment IDs (all 0s for single sentence)
        """
        # Tokenize
        caption_tokens = self.tokenizer.tokenize(caption)
        
        # Truncate
        max_caption_len = self.max_words - 2  # Reserve for [CLS] and [SEP]
        if len(caption_tokens) > max_caption_len:
            caption_tokens = caption_tokens[:max_caption_len]
        
        # Add special tokens
        tokens = ['[CLS]'] + caption_tokens + ['[SEP]']
        
        # Convert to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        
        # Padding
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
        }
    
    def _get_video_features(self, video_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess video features
        
        Returns:
            video: Normalized video features (max_frames, feature_dim)
            video_mask: Valid frame mask (max_frames,)
        """
        # Load features
        video_features = self.feature_dict[video_id]
        
        # Convert to tensor if needed
        if not isinstance(video_features, torch.Tensor):
            video_features = torch.from_numpy(video_features)
        
        # Normalize and pad
        video, video_mask = self.video_transform(video_features, self.max_frames)
        
        return video, video_mask
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Returns dict with keys:
            - video: Video features
            - video_mask: Valid frame mask
            - input_ids: Text token IDs
            - attention_mask: Text attention mask
            - token_type_ids: Text segment IDs
            - input_caption_ids: Decoder input IDs
            - output_caption_ids: Decoder target IDs
            - decoder_mask: Decoder mask
            
            If is_pretrain=True, also includes:
            - masked_input_ids: Masked text for MLM
            - mlm_labels: Labels for MLM
            - masked_video: Masked video for MFM
            - mfm_labels: Labels for MFM
        """
        sample = self.samples[idx]
        video_id = sample['video_id']
        caption = sample['caption']
        
        # Get video features
        video, video_mask = self._get_video_features(video_id)
        
        # Get text features (for encoder)
        text_features = self._get_text_features(caption)
        
        # Get caption features (for decoder)
        caption_features = self.caption_transform(caption)
        
        # Build output dict (match model's expected parameter names)
        output = {
            # 'video_id': video_id,  # Not needed by model, remove to avoid error
            'video': video,
            'video_mask': video_mask,
            'input_ids': text_features['input_ids'],
            'attention_mask': text_features['attention_mask'],
            'token_type_ids': text_features['token_type_ids'],
            'input_caption_ids': caption_features['decoder_input_ids'],  # Model expects this name
            'output_caption_ids': caption_features['decoder_target_ids'],  # Model expects this name
            'decoder_mask': caption_features['decoder_mask'],
        }
        
        # Add masking for pretraining
        if self.is_pretrain:
            # MLM masking
            mlm_output = self.mask_tokens(text_features['input_ids'])
            output['masked_input_ids'] = mlm_output['masked_input_ids']
            output['mlm_labels'] = mlm_output['labels']  # Model expects this name
            
            # MFM masking
            mfm_output = self.mask_frames(video, video_mask)
            output['masked_video'] = mfm_output['masked_video']
            output['mfm_labels'] = mfm_output['labels']  # Model expects this name
        
        return output


class MSRVTTPretrainDataset(MSRVTTCaptionDataset):
    """
    MSRVTT dataset for pretraining with MLM and MFM
    
    This is just a convenience wrapper that sets is_pretrain=True
    """
    
    def __init__(
        self,
        data_path: str,
        features_path: str,
        tokenizer: BertTokenizer,
        max_words: int = 20,
        max_frames: int = 100,
        split_type: str = "train",
    ):
        super().__init__(
            data_path=data_path,
            features_path=features_path,
            tokenizer=tokenizer,
            max_words=max_words,
            max_frames=max_frames,
            split_type=split_type,
            is_pretrain=True,  # Always True for pretrain dataset
        )


def create_dataloader(
    config,
    tokenizer: BertTokenizer,
    split: str = "train",
    is_pretrain: bool = False,
    shuffle: bool = None,
    drop_last: bool = None,
) -> DataLoader:
    """
    Create DataLoader for MSRVTT dataset
    
    Args:
        config: Config object with data settings
        tokenizer: BERT tokenizer
        split: "train", "val", or "test"
        is_pretrain: Whether this is for pretraining
        shuffle: Whether to shuffle (default: True for train, False for val/test)
        drop_last: Whether to drop last incomplete batch (default: True for train, False for val/test)
    
    Returns:
        DataLoader instance
    """
    # Create dataset
    if is_pretrain:
        dataset = MSRVTTPretrainDataset(
            data_path=config.data.data_path,
            features_path=config.data.features_path,
            tokenizer=tokenizer,
            max_words=config.data.max_words,
            max_frames=config.data.max_frames,
            split_type=split,
        )
    else:
        dataset = MSRVTTCaptionDataset(
            data_path=config.data.data_path,
            features_path=config.data.features_path,
            tokenizer=tokenizer,
            max_words=config.data.max_words,
            max_frames=config.data.max_frames,
            split_type=split,
            is_pretrain=False,
        )
    
    # Default shuffle and drop_last behavior
    if shuffle is None:
        shuffle = (split == "train")
    if drop_last is None:
        drop_last = (split == "train")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size if split == "train" else config.training.batch_size_val,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    
    return dataloader
