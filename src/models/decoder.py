"""Caption Decoder module - Following source UniVL structure for checkpoint compatibility"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math
import copy


class DecoderEmbeddings(nn.Module):
    """
    Decoder embeddings module
    Structure matches checkpoint: decoder.embeddings.*
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        hidden_dropout_prob: float,
        word_embeddings_weight: Optional[torch.Tensor] = None,
        position_embeddings_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        # Word embeddings (share with BERT if provided)
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        if word_embeddings_weight is not None:
            self.word_embeddings.weight = word_embeddings_weight
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        if position_embeddings_weight is not None:
            self.position_embeddings.weight = position_embeddings_weight
        
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        embeddings = word_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
    ):
        super().__init__()
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"hidden_size must be divisible by num_attention_heads")
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mixed_query_layer = self.query(q)
        mixed_key_layer = self.key(k)
        mixed_value_layer = self.value(v)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer, attention_scores


class SelfOutput(nn.Module):
    """Output projection after attention"""
    def __init__(self, hidden_size: int, hidden_dropout_prob: float):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DecoderAttention(nn.Module):
    """
    Attention wrapper combining att + output
    Structure matches: decoder.decoder.layer.X.slf_attn or enc_attn
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
    ):
        super().__init__()
        self.att = MultiHeadAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        att_output, attention_probs = self.att(q, k, v, attention_mask)
        attention_output = self.output(att_output, q)
        return attention_output, attention_probs


class BertIntermediate(nn.Module):
    """Feed-forward intermediate layer"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """Feed-forward output layer"""
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_dropout_prob: float):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DecoderLayer(nn.Module):
    """
    Single decoder layer
    Structure matches checkpoint: decoder.decoder.layer.X
    - slf_attn: self-attention
    - enc_attn: encoder-decoder attention (cross-attention)
    - intermediate: feed-forward intermediate
    - output: feed-forward output
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
    ):
        super().__init__()
        
        # Self-attention (matches: decoder.decoder.layer.X.slf_attn)
        self.slf_attn = DecoderAttention(
            hidden_size, num_attention_heads, hidden_dropout_prob, attention_probs_dropout_prob
        )
        
        # Cross-attention (matches: decoder.decoder.layer.X.enc_attn)
        self.enc_attn = DecoderAttention(
            hidden_size, num_attention_heads, hidden_dropout_prob, attention_probs_dropout_prob
        )
        
        # Feed-forward
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(hidden_size, intermediate_size, hidden_dropout_prob)
    
    def forward(
        self,
        dec_input: torch.Tensor,
        enc_output: torch.Tensor,
        slf_attn_mask: torch.Tensor,
        dec_enc_attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        slf_output, _ = self.slf_attn(dec_input, dec_input, dec_input, slf_attn_mask)
        
        # Cross-attention
        dec_output, dec_att_scores = self.enc_attn(slf_output, enc_output, enc_output, dec_enc_attn_mask)
        
        # Feed-forward
        intermediate_output = self.intermediate(dec_output)
        dec_output = self.output(intermediate_output, dec_output)
        
        return dec_output, dec_att_scores


class Decoder(nn.Module):
    """
    Decoder stack containing multiple DecoderLayers
    Structure matches checkpoint: decoder.decoder.layer (ModuleList)
    """
    def __init__(
        self,
        num_decoder_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
    ):
        super().__init__()
        
        layer = DecoderLayer(
            hidden_size, num_attention_heads, intermediate_size,
            hidden_dropout_prob, attention_probs_dropout_prob
        )
        # IMPORTANT: Must be named "layer" to match checkpoint!
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_decoder_layers)])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_outs: torch.Tensor,
        self_attn_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        output_all_encoded_layers: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        all_encoder_layers = []
        all_dec_att_probs = []
        
        for layer_module in self.layer:
            hidden_states, dec_att_scores = layer_module(
                hidden_states, encoder_outs, self_attn_mask, attention_mask
            )
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                all_dec_att_probs.append(dec_att_scores)
        
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_dec_att_probs.append(dec_att_scores)
        
        return all_encoder_layers, all_dec_att_probs


class PredictionHeadTransform(nn.Module):
    """Transform layer before LM prediction"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class LMPredictionHead(nn.Module):
    """
    Language modeling prediction head
    Structure matches: decoder.classifier.cls.predictions
    """
    def __init__(self, hidden_size: int, vocab_size: int, embedding_weights: torch.Tensor):
        super().__init__()
        self.transform = PredictionHeadTransform(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.decoder.weight = embedding_weights
        self.bias = nn.Parameter(torch.zeros(vocab_size))
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class DecoderClassifier(nn.Module):
    """
    Decoder classifier head
    Structure matches checkpoint: decoder.classifier.cls.predictions
    """
    def __init__(self, hidden_size: int, vocab_size: int, embedding_weights: torch.Tensor):
        super().__init__()
        # IMPORTANT: Must wrap in ModuleDict with key "cls" then "predictions"
        self.cls = nn.ModuleDict({
            'predictions': LMPredictionHead(hidden_size, vocab_size, embedding_weights)
        })
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.cls['predictions'](hidden_states)


class CaptionDecoder(nn.Module):
    """
    Caption Decoder - Refactored to match source UniVL structure
    
    Checkpoint key mapping (NOW MATCHES DIRECTLY):
    - decoder.embeddings.* → self.embeddings
    - decoder.decoder.layer.* → self.decoder.layer
    - decoder.classifier.* → self.classifier
    """
    
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_decoder_layers: int = 3,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 128,
        bert_word_embeddings_weight: Optional[torch.Tensor] = None,
        bert_position_embeddings_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Embeddings: decoder.embeddings.*
        self.embeddings = DecoderEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            word_embeddings_weight=bert_word_embeddings_weight,
            position_embeddings_weight=bert_position_embeddings_weight,
        )
        
        # Decoder layers: decoder.decoder.layer.*
        self.decoder = Decoder(
            num_decoder_layers=num_decoder_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
        
        # Classifier: decoder.classifier.*
        self.classifier = DecoderClassifier(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            embedding_weights=self.embeddings.word_embeddings.weight,
        )
    
    def forward(
        self,
        input_caption_ids: torch.Tensor,
        encoder_outs: torch.Tensor,
        answer_mask: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for training (teacher forcing)
        
        Args:
            input_caption_ids: Caption tokens (batch_size, max_words)
            encoder_outs: Encoder outputs (batch_size, seq_len, hidden_size)
            answer_mask: Caption attention mask (batch_size, max_words)
            encoder_mask: Encoder attention mask (batch_size, seq_len)
        
        Returns:
            logits: Vocabulary logits (batch_size, max_words, vocab_size)
        """
        # Get embeddings
        embedding_output = self.embeddings(input_caption_ids)
        batch_size, seq_length = input_caption_ids.size()
        
        # Prepare encoder mask for cross-attention
        extended_encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, src_len)
        extended_encoder_mask = extended_encoder_mask.to(dtype=embedding_output.dtype)
        extended_encoder_mask = (1.0 - extended_encoder_mask) * -10000.0
        
        # Prepare causal mask for self-attention
        extended_answer_mask = answer_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, tgt_len)
        extended_answer_mask = extended_answer_mask.to(dtype=embedding_output.dtype)
        
        # Create causal (triangular) mask
        subsequent_mask = torch.triu(
            torch.ones((seq_length, seq_length), device=input_caption_ids.device, dtype=embedding_output.dtype),
            diagonal=1
        )
        self_attn_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1)
        slf_attn_mask = ((1.0 - extended_answer_mask) + self_attn_mask).gt(0).to(dtype=embedding_output.dtype)
        slf_attn_mask = slf_attn_mask * -10000.0
        
        # Decoder forward
        decoded_layers, _ = self.decoder(
            embedding_output,
            encoder_outs,
            slf_attn_mask,
            extended_encoder_mask,
            output_all_encoded_layers=True,
        )
        
        # Get last layer output
        sequence_output = decoded_layers[-1]
        
        # Classifier
        cls_scores = self.classifier(sequence_output)
        
        return cls_scores
    
    def generate(
        self,
        encoder_outs: torch.Tensor,
        encoder_mask: torch.Tensor,
        max_length: int = 20,
        bos_token_id: int = 101,
        eos_token_id: int = 102,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """
        Greedy generation (for simple inference)
        
        Args:
            encoder_outs: Encoder outputs (batch_size, seq_len, hidden_size)
            encoder_mask: Encoder mask (batch_size, seq_len)
            max_length: Maximum generation length
            bos_token_id: Begin-of-sequence token ID
            eos_token_id: End-of-sequence token ID
            pad_token_id: Padding token ID
        
        Returns:
            generated_ids: Generated token IDs (batch_size, max_length)
        """
        batch_size = encoder_outs.size(0)
        device = encoder_outs.device
        
        # Start with BOS token
        generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # Create answer mask
            answer_mask = torch.ones(batch_size, generated.size(1), device=device)
            
            # Forward pass
            logits = self.forward(generated, encoder_outs, answer_mask, encoder_mask)
            
            # Get next token (greedy)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have EOS
            if (next_token == eos_token_id).all():
                break
        
        return generated
