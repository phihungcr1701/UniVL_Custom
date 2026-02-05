"""Caption Decoder module for generating captions"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class CaptionDecoder(nn.Module):
    """
    Transformer decoder for caption generation
    Conditions on encoder outputs (cross-modal features) to generate captions
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
        """
        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden dimension
            num_decoder_layers: Number of decoder layers
            num_attention_heads: Number of attention heads
            intermediate_size: FFN intermediate dimension
            hidden_dropout_prob: Dropout probability
            attention_probs_dropout_prob: Attention dropout
            max_position_embeddings: Max caption length
            bert_word_embeddings_weight: Pretrained word embeddings from BERT
            bert_position_embeddings_weight: Pretrained position embeddings from BERT
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_decoder_layers = num_decoder_layers
        self.vocab_size = vocab_size
        
        # Embeddings (share with BERT encoder if provided)
        if bert_word_embeddings_weight is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(
                bert_word_embeddings_weight, freeze=False
            )
        else:
            self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        if bert_position_embeddings_weight is not None:
            self.position_embeddings = nn.Embedding.from_pretrained(
                bert_position_embeddings_weight, freeze=False
            )
        else:
            self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
            )
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights with word embeddings
        self.lm_head.weight = self.word_embeddings.weight
    
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
        batch_size, seq_length = input_caption_ids.size()
        
        # Create embeddings
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_caption_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        word_embeds = self.word_embeddings(input_caption_ids)
        position_embeds = self.position_embeddings(position_ids)
        embeddings = word_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Prepare masks
        # Self-attention mask (causal mask for autoregressive)
        causal_mask = self._generate_causal_mask(seq_length, input_caption_ids.device)
        answer_mask_expanded = answer_mask.unsqueeze(1).unsqueeze(2)
        self_attn_mask = causal_mask * answer_mask_expanded
        self_attn_mask = (1.0 - self_attn_mask) * -10000.0
        
        # Cross-attention mask
        cross_attn_mask = encoder_mask.unsqueeze(1).unsqueeze(2)
        cross_attn_mask = (1.0 - cross_attn_mask) * -10000.0
        
        # Decoder layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                encoder_outs,
                self_attn_mask,
                cross_attn_mask,
            )
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(size, size, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)
    
    def generate(
        self,
        encoder_outs: torch.Tensor,
        encoder_mask: torch.Tensor,
        max_length: int = 20,
        bos_token_id: int = 101,  # [CLS]
        eos_token_id: int = 102,  # [SEP]
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


class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention and cross-attention"""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
    ):
        super().__init__()
        
        # Self-attention
        self.self_attention = MultiHeadAttention(
            hidden_size, num_attention_heads,
            hidden_dropout_prob, attention_probs_dropout_prob
        )
        
        # Cross-attention
        self.cross_attention = MultiHeadAttention(
            hidden_size, num_attention_heads,
            hidden_dropout_prob, attention_probs_dropout_prob,
            is_cross_attention=True,
        )
        
        # Feed-forward
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = DecoderOutput(hidden_size, intermediate_size, hidden_dropout_prob)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_outputs: torch.Tensor,
        self_attn_mask: torch.Tensor,
        cross_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention
        self_attn_output = self.self_attention(hidden_states, hidden_states, self_attn_mask)
        
        # Cross-attention
        cross_attn_output = self.cross_attention(self_attn_output, encoder_outputs, cross_attn_mask)
        
        # Feed-forward
        intermediate_output = self.intermediate(cross_attn_output)
        intermediate_output = F.gelu(intermediate_output)
        layer_output = self.output(intermediate_output, cross_attn_output)
        
        return layer_output


class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        is_cross_attention: bool = False,
    ):
        super().__init__()
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.is_cross_attention = is_cross_attention
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.output_dropout = nn.Dropout(hidden_dropout_prob)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(key_value_states)
        mixed_value_layer = self.value(key_value_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        
        # Attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Context layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Output projection
        attention_output = self.dense(context_layer)
        attention_output = self.output_dropout(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)
        
        return attention_output


class DecoderOutput(nn.Module):
    """Output layer for decoder"""
    
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
