"""UniVL Model - Unified Video-Language model for video captioning"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .text_encoder import BertModel
from .visual_encoder import VisualModel
from .cross_encoder import CrossModel
from .decoder import CaptionDecoder
from .beam import Beam


class UniVLModel(nn.Module):
    """
    UniVL model for video captioning
    Supports both pretrain (MLM/MFM) and fine-tuning (caption only) modes
    """
    
    def __init__(self, config):
        """
        Args:
            config: UniVLConfig object with model, training, and data settings
        """
        super().__init__()
        
        self.config = config
        self.model_config = config.model
        self.pretrain_mode = config.model.pretrain_mode
        
        # ===== Encoders =====
        # Text encoder (BERT) - matches checkpoint: bert.*
        self.bert = BertModel(
            vocab_size=self.model_config.vocab_size,
            hidden_size=self.model_config.hidden_size,
            num_hidden_layers=self.model_config.text_num_hidden_layers,
            hidden_dropout_prob=self.model_config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.model_config.attention_probs_dropout_prob,
        )
        
        # Visual encoder - matches checkpoint: visual.*
        self.visual = VisualModel(
            video_dim=self.model_config.video_dim,
            hidden_size=self.model_config.hidden_size,
            num_hidden_layers=self.model_config.visual_num_hidden_layers,
            hidden_dropout_prob=self.model_config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.model_config.attention_probs_dropout_prob,
            max_position_embeddings=512,  # Fixed: must match checkpoint (not config.max_frames)
        )
        
        # Cross encoder - matches checkpoint: cross.*
        self.cross = CrossModel(
            hidden_size=self.model_config.hidden_size,
            num_hidden_layers=self.model_config.cross_num_hidden_layers,
            hidden_dropout_prob=self.model_config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.model_config.attention_probs_dropout_prob,
            max_position_embeddings=1024,  # Fixed: must match checkpoint (source uses 1024)
        )
        
        # ===== Decoder =====
        # Get word embeddings from BERT for sharing
        bert_word_embeddings = self.bert.embeddings.word_embeddings.weight
        bert_position_embeddings = self.bert.embeddings.position_embeddings.weight
        
        self.decoder = CaptionDecoder(
            vocab_size=self.model_config.vocab_size,
            hidden_size=self.model_config.hidden_size,
            num_decoder_layers=self.model_config.decoder_num_hidden_layers,
            hidden_dropout_prob=self.model_config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.model_config.attention_probs_dropout_prob,
            max_position_embeddings=self.model_config.max_words,
            bert_word_embeddings_weight=bert_word_embeddings,
            bert_position_embeddings_weight=bert_position_embeddings,
        )
        
        # ===== Pretrain heads (optional) =====
        if self.pretrain_mode:
            # MLM head (predict masked tokens)
            self.mlm_head = nn.Linear(self.model_config.hidden_size, self.model_config.vocab_size)
            self.mlm_head.weight = bert_word_embeddings  # Tie weights
            
            # MFM head (predict masked frames via contrastive learning)
            # Projects visual features for frame prediction
            self.mfm_head = nn.Linear(self.model_config.hidden_size, self.model_config.video_dim)
        
        # Loss functions
        self.caption_loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        if self.pretrain_mode:
            self.mlm_loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        video: torch.Tensor,
        video_mask: torch.Tensor,
        input_caption_ids: torch.Tensor,
        decoder_mask: torch.Tensor,
        output_caption_ids: torch.Tensor,
        masked_input_ids: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.Tensor] = None,
        masked_video: Optional[torch.Tensor] = None,
        mfm_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Text token IDs (batch_size, max_words)
            attention_mask: Text attention mask (batch_size, max_words)
            token_type_ids: Token type IDs (batch_size, max_words)
            video: Video features (batch_size, max_frames, video_dim)
            video_mask: Video mask (batch_size, max_frames)
            input_caption_ids: Caption input IDs (batch_size, max_words)
            decoder_mask: Decoder attention mask (batch_size, max_words)
            output_caption_ids: Caption target IDs (batch_size, max_words)
            masked_input_ids: Masked text for MLM (optional)
            mlm_labels: MLM labels (optional)
            masked_video: Masked video for MFM (optional)
            mfm_labels: MFM labels (optional)
        
        Returns:
            Dictionary with losses and outputs
        """
        outputs = {}
        total_loss = 0.0
        
        # Determine which inputs to use (masked for pretrain, regular for fine-tuning)
        if self.pretrain_mode and self.training and masked_input_ids is not None:
            text_input = masked_input_ids
            video_input = masked_video
        else:
            text_input = input_ids
            video_input = video
        
        # ===== Encode =====
        # Text encoding
        text_hidden, _ = self.bert(
            text_input,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_all_encoded_layers=True,
        )
        sequence_output = text_hidden[-1]  # Last layer
        
        # Visual encoding
        visual_hidden, _ = self.visual(
            video_input,
            attention_mask=video_mask,
            output_all_encoded_layers=True,
        )
        visual_output = visual_hidden[-1]  # Last layer
        
        # ===== Cross-modal fusion =====
        # Concatenate text and video
        concat_features = torch.cat([sequence_output, visual_output], dim=1)
        concat_mask = torch.cat([attention_mask, video_mask], dim=1)
        
        # Token type: 0 for text, 1 for video
        # IMPORTANT: Must use .long() for embedding layer
        text_type = torch.zeros_like(attention_mask, dtype=torch.long)
        video_type = torch.ones_like(video_mask, dtype=torch.long)
        concat_type = torch.cat([text_type, video_type], dim=1)
        
        # Cross encoding
        cross_hidden, cross_pooled = self.cross(
            concat_features,
            concat_type=concat_type,
            attention_mask=concat_mask,
            output_all_encoded_layers=True,
        )
        cross_output = cross_hidden[-1]  # Last layer
        
        # ===== Caption generation =====
        # Decoder forward pass
        decoder_logits = self.decoder(
            input_caption_ids,
            cross_output,
            decoder_mask,
            concat_mask,
        )
        
        # Caption loss
        caption_loss = self.caption_loss_fct(
            decoder_logits.view(-1, self.model_config.vocab_size),
            output_caption_ids.view(-1),
        )
        total_loss += caption_loss
        outputs['caption_loss'] = caption_loss
        outputs['decoder_logits'] = decoder_logits
        
        # ===== Pretrain losses (if enabled) =====
        if self.pretrain_mode and self.training:
            # Split cross output back to text and video
            text_cross_output, visual_cross_output = torch.split(
                cross_output,
                [attention_mask.size(1), video_mask.size(1)],
                dim=1,
            )
            
            # MLM loss
            if mlm_labels is not None:
                mlm_logits = self.mlm_head(text_cross_output)
                mlm_loss = self.mlm_loss_fct(
                    mlm_logits.view(-1, self.model_config.vocab_size),
                    mlm_labels.view(-1),
                )
                total_loss += mlm_loss
                outputs['mlm_loss'] = mlm_loss
            
            # MFM loss (contrastive)
            if mfm_labels is not None:
                mfm_loss = self._calculate_mfm_loss(
                    visual_cross_output,
                    video,
                    video_mask,
                    mfm_labels,
                )
                total_loss += mfm_loss
                outputs['mfm_loss'] = mfm_loss
        
        outputs['loss'] = total_loss
        return outputs
    
    def _calculate_mfm_loss(
        self,
        visual_output: torch.Tensor,
        video: torch.Tensor,
        video_mask: torch.Tensor,
        video_labels_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate Masked Frame Modeling loss using contrastive learning
        
        Args:
            visual_output: Visual encoder output (batch_size, max_frames, hidden_size)
            video: Original video features (batch_size, max_frames, video_dim)
            video_mask: Video mask (batch_size, max_frames)
            video_labels_index: Indices of masked frames (batch_size, max_frames)
        
        Returns:
            MFM loss scalar
        """
        # Project visual output back to video dimension
        mfm_scores = self.mfm_head(visual_output)
        mfm_scores_flat = mfm_scores.view(-1, mfm_scores.shape[-1])
        
        # Transpose video for matrix multiplication
        video_transposed = video.permute(2, 0, 1)  # (video_dim, batch_size, max_frames)
        video_transposed = video_transposed.reshape(video_transposed.shape[0], -1)  # (video_dim, batch*frames)
        
        # Compute similarity matrix
        logits_matrix = torch.mm(mfm_scores_flat, video_transposed)  # (batch*frames, batch*frames)
        
        # Mask invalid positions
        video_mask_float = video_mask.float()
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1.0 - mask_matrix) * -1e8
        
        # Compute loss (diagonal elements should be highest)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        diagonal_log_probs = torch.diag(log_probs)
        nce_loss = -diagonal_log_probs
        
        # Only compute loss for masked frames
        valid_mask = (video_labels_index != -1).view(-1)
        nce_loss = nce_loss[valid_mask]
        
        if nce_loss.numel() > 0:
            return nce_loss.mean()
        else:
            return torch.tensor(0.0, device=nce_loss.device)
    
    def generate_caption(
        self,
        video: torch.Tensor,
        video_mask: torch.Tensor,
        max_length: int = 20,
        beam_size: int = 5,
        bos_token_id: int = 101,
        eos_token_id: int = 102,
        tokenizer=None,
    ) -> torch.Tensor:
        """
        Generate caption for video using beam search (following source UniVL implementation)
        
        Args:
            video: Video features (batch_size, max_frames, video_dim)
            video_mask: Video mask (batch_size, max_frames)
            max_length: Maximum caption length
            beam_size: Beam size for beam search (1 = greedy)
            bos_token_id: Begin-of-sequence token
            eos_token_id: End-of-sequence token
            tokenizer: Tokenizer for beam search
        
        Returns:
            Generated caption token IDs (batch_size, max_length)
        """
        batch_size = video.size(0)
        device = video.device
        
        # Create empty text input (for video-only captioning)
        input_ids = torch.full((batch_size, 2), bos_token_id, dtype=torch.long, device=device)
        input_ids[:, 1] = eos_token_id
        attention_mask = torch.ones(batch_size, 2, dtype=torch.long, device=device)
        token_type_ids = torch.zeros(batch_size, 2, dtype=torch.long, device=device)
        
        # Encode
        text_hidden, _ = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_all_encoded_layers=True,
        )
        sequence_output = text_hidden[-1]
        
        visual_hidden, _ = self.visual(
            video,
            attention_mask=video_mask,
            output_all_encoded_layers=True,
        )
        visual_output = visual_hidden[-1]
        
        # Cross encoding
        concat_features = torch.cat([sequence_output, visual_output], dim=1)
        concat_mask = torch.cat([attention_mask, video_mask], dim=1)
        # IMPORTANT: Must use .long() for embedding layer
        text_type = torch.zeros_like(attention_mask, dtype=torch.long)
        video_type = torch.ones_like(video_mask, dtype=torch.long)
        concat_type = torch.cat([text_type, video_type], dim=1)
        
        cross_hidden, _ = self.cross(
            concat_features,
            concat_type=concat_type,
            attention_mask=concat_mask,
            output_all_encoded_layers=True,
        )
        cross_output = cross_hidden[-1]
        
        # Generate caption
        if beam_size == 1:
            # Greedy decoding
            generated = self.decoder.generate(
                cross_output,
                concat_mask,
                max_length=max_length,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
            )
        else:
            # Beam search (following source UniVL implementation)
            generated = self._beam_search(
                cross_output,
                concat_mask,
                max_length=max_length,
                beam_size=beam_size,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                tokenizer=tokenizer,
            )
        
        return generated
    
    def _beam_search(
        self,
        cross_output: torch.Tensor,
        cross_mask: torch.Tensor,
        max_length: int,
        beam_size: int,
        bos_token_id: int,
        eos_token_id: int,
        tokenizer=None,
    ) -> torch.Tensor:
        """
        Beam search implementation (following source UniVL)
        
        Args:
            cross_output: Cross encoder output (batch_size, seq_len, hidden_size)
            cross_mask: Cross attention mask (batch_size, seq_len)
            max_length: Maximum generation length
            beam_size: Beam size
            bos_token_id: BOS token ID
            eos_token_id: EOS token ID
            tokenizer: Tokenizer for beam

        Returns:
            Generated sequences (batch_size, max_length)
        """
        device = cross_output.device
        batch_size = cross_output.size(0)
        
        # Repeat cross_output and mask for beam search
        # Shape: (batch_size * beam_size, seq_len, hidden_size)
        cross_output_repeat = cross_output.repeat(1, beam_size, 1).view(
            batch_size * beam_size, cross_output.size(1), cross_output.size(2)
        )
        cross_mask_repeat = cross_mask.repeat(1, beam_size).view(
            batch_size * beam_size, cross_mask.size(1)
        )
        
        # Initialize beams for each instance
        inst_dec_beams = [Beam(beam_size, device=device, tokenizer=tokenizer) for _ in range(batch_size)]
        
        # Active instances
        active_inst_idx_list = list(range(batch_size))
        inst_idx_to_position_map = {idx: pos for pos, idx in enumerate(active_inst_idx_list)}
        
        # Decode step by step
        for len_dec_seq in range(1, max_length + 1):
            # Get current decoder inputs from beams
            dec_partial_seq = self._prepare_beam_dec_seq(inst_dec_beams, len_dec_seq, device)
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            
            # Get active cross outputs
            n_active_inst = len(active_inst_idx_list)
            active_cross_output = self._gather_active_data(
                cross_output_repeat, inst_idx_to_position_map, active_inst_idx_list, batch_size, beam_size
            )
            active_cross_mask = self._gather_active_data(
                cross_mask_repeat, inst_idx_to_position_map, active_inst_idx_list, batch_size, beam_size
            )
            
            # Predict word probabilities
            word_prob = self._predict_word(
                dec_partial_seq, active_cross_output, active_cross_mask, n_active_inst, beam_size
            )
            
            # Update beams
            active_inst_idx_list = self._collect_active_inst_idx(
                inst_dec_beams, word_prob, inst_idx_to_position_map
            )
            
            if not active_inst_idx_list:
                break  # All beams finished
            
            # Update inst_idx_to_position_map
            inst_idx_to_position_map = {idx: pos for pos, idx in enumerate(active_inst_idx_list)}
        
        # Collect hypothesis
        batch_hyp = []
        for inst_idx in range(batch_size):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            best_hyp = inst_dec_beams[inst_idx].get_hypothesis(tail_idxs[0])
            batch_hyp.append(best_hyp)
        
        # Convert to tensor and pad
        generated = torch.full((batch_size, max_length), 0, dtype=torch.long, device=device)
        for i, hyp in enumerate(batch_hyp):
            generated[i, :len(hyp)] = torch.tensor(hyp[:max_length], dtype=torch.long, device=device)
        
        return generated
    
    def _prepare_beam_dec_seq(self, inst_dec_beams, len_dec_seq, device):
        """Prepare decoder input sequences from beams"""
        dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_partial_seq = torch.stack(dec_partial_seq).to(device)
        return dec_partial_seq
    
    def _gather_active_data(self, data, inst_idx_to_position_map, active_inst_idx_list, batch_size, beam_size):
        """Gather data for active instances"""
        n_active = len(active_inst_idx_list)
        new_shape = (n_active * beam_size,) + data.shape[1:]
        active_data = data.new_zeros(new_shape)
        
        for beam_idx, inst_idx in enumerate(active_inst_idx_list):
            old_pos = inst_idx_to_position_map[inst_idx]
            start_old = old_pos * beam_size
            start_new = beam_idx * beam_size
            active_data[start_new:start_new + beam_size] = data[start_old:start_old + beam_size]
        
        return active_data
    
    def _predict_word(self, dec_seq, cross_output, cross_mask, n_active_inst, beam_size):
        """Predict word probabilities"""
        dec_mask = torch.ones(dec_seq.size(), dtype=torch.long, device=dec_seq.device)
        
        # Forward through decoder
        dec_logits = self.decoder(
            dec_seq,
            cross_output,
            dec_mask,
            cross_mask,
        )
        
        # Get last token logits
        dec_output = dec_logits[:, -1, :]
        word_prob = F.log_softmax(dec_output, dim=1)
        word_prob = word_prob.view(n_active_inst, beam_size, -1)
        
        return word_prob
    
    def _collect_active_inst_idx(self, inst_beams, word_prob, inst_idx_to_position_map):
        """Collect indices of active instances"""
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
            if not is_inst_complete:
                active_inst_idx_list.append(inst_idx)
        
        return active_inst_idx_list

