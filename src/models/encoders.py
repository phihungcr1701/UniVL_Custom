"""Encoder modules: Text, Visual, and Cross encoders"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from transformers import BertModel, BertConfig


class TextEncoder(nn.Module):
    """Text encoder using HuggingFace BERT"""
    
    def __init__(
        self,
        pretrained_model: str = "bert-base-uncased",
        num_hidden_layers: int = 12,
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.1,
    ):
        """
        Args:
            pretrained_model: HuggingFace model name
            num_hidden_layers: Number of transformer layers
            hidden_size: Hidden dimension
            hidden_dropout_prob: Dropout probability
        """
        super().__init__()
        
        # Load pretrained BERT config and modify
        config = BertConfig.from_pretrained(pretrained_model)
        config.num_hidden_layers = num_hidden_layers
        config.hidden_size = hidden_size
        config.hidden_dropout_prob = hidden_dropout_prob
        
        # Load BERT model
        self.bert = BertModel.from_pretrained(
            pretrained_model,
            config=config,
            ignore_mismatched_sizes=True,
        )
        
        self.hidden_size = hidden_size
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        output_all_encoded_layers: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs (batch_size, max_words)
            attention_mask: Attention mask (batch_size, max_words)
            token_type_ids: Token type IDs (batch_size, max_words)
            output_all_encoded_layers: Return all layers or just last one
        
        Returns:
            sequence_output: Last hidden states (batch_size, max_words, hidden_size)
            pooled_output: Pooled output (batch_size, hidden_size)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_all_encoded_layers,
        )
        
        if output_all_encoded_layers:
            # Return all hidden states
            return outputs.hidden_states, outputs.pooler_output
        else:
            # Return last hidden state
            return outputs.last_hidden_state, outputs.pooler_output


class VisualEncoder(nn.Module):
    """Visual encoder for video features (adapted from UniVL)"""
    
    def __init__(
        self,
        video_dim: int = 1024,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
    ):
        """
        Args:
            video_dim: Input video feature dimension (e.g., 1024 for S3D)
            hidden_size: Hidden dimension
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            intermediate_size: FFN intermediate dimension
            hidden_dropout_prob: Dropout probability
            attention_probs_dropout_prob: Attention dropout
            max_position_embeddings: Max sequence length
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        
        # Embeddings: project video features to hidden_size
        self.embeddings = VisualEmbeddings(
            video_dim=video_dim,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
        )
        
        # Transformer encoder layers
        self.encoder = nn.ModuleList([
            VisualLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
            )
            for _ in range(num_hidden_layers)
        ])
    
    def forward(
        self,
        video: torch.Tensor,
        video_mask: torch.Tensor,
        output_all_encoded_layers: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            video: Video features (batch_size, max_frames, video_dim)
            video_mask: Video mask (batch_size, max_frames)
            output_all_encoded_layers: Return all layers or just last one
        
        Returns:
            encoded_layers: Hidden states from all/last layer
            pooled_output: Always None (for compatibility)
        """
        # Prepare attention mask (1 -> 0, 0 -> -10000)
        extended_video_mask = video_mask.unsqueeze(1).unsqueeze(2)
        extended_video_mask = (1.0 - extended_video_mask) * -10000.0
        
        # Embeddings
        embedding_output = self.embeddings(video)
        
        # Encoder layers
        all_encoder_layers = []
        hidden_states = embedding_output
        
        for layer_module in self.encoder:
            hidden_states = layer_module(hidden_states, extended_video_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        
        return all_encoder_layers, None


class VisualEmbeddings(nn.Module):
    """Visual embeddings with position encoding"""
    
    def __init__(
        self,
        video_dim: int,
        hidden_size: int,
        max_position_embeddings: int,
        hidden_dropout_prob: float,
    ):
        super().__init__()
        
        # Project video features to hidden_size
        self.word_embeddings = nn.Linear(video_dim, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = input_embeddings.size()
        
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Project and add position embeddings
        words_embeddings = self.word_embeddings(input_embeddings)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class VisualLayer(nn.Module):
    """Visual transformer layer"""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
    ):
        super().__init__()
        
        self.attention = VisualAttention(
            hidden_size, num_attention_heads,
            hidden_dropout_prob, attention_probs_dropout_prob
        )
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = VisualOutput(hidden_size, intermediate_size, hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = nn.functional.gelu(intermediate_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class VisualAttention(nn.Module):
    """Multi-head self-attention for visual encoder"""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
    ):
        super().__init__()
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
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
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        attention_scores = attention_scores + attention_mask
        
        # Attention probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
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


class VisualOutput(nn.Module):
    """Output layer for visual transformer"""
    
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


class CrossEncoder(nn.Module):
    """Cross-modal encoder for fusing text and video"""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
    ):
        """
        Args:
            hidden_size: Hidden dimension
            num_hidden_layers: Number of cross-modal layers
            num_attention_heads: Number of attention heads
            intermediate_size: FFN intermediate dimension
            hidden_dropout_prob: Dropout probability
            attention_probs_dropout_prob: Attention dropout
            max_position_embeddings: Max sequence length (text + video)
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Token type embeddings (0 for text, 1 for video)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        
        # Transformer layers
        self.encoder = nn.ModuleList([
            VisualLayer(  # Reuse VisualLayer architecture
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Pooler for getting sequence representation
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()
    
    def forward(
        self,
        concat_features: torch.Tensor,
        concat_type: torch.Tensor,
        concat_mask: torch.Tensor,
        output_all_encoded_layers: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            concat_features: Concatenated text + video features (batch_size, seq_len, hidden_size)
            concat_type: Token type IDs (0 for text, 1 for video)
            concat_mask: Attention mask (batch_size, seq_len)
            output_all_encoded_layers: Return all layers or just last one
        
        Returns:
            encoded_layers: Hidden states
            pooled_output: Pooled representation
        """
        # Add token type embeddings
        token_type_embeds = self.token_type_embeddings(concat_type)
        hidden_states = concat_features + token_type_embeds
        
        # Prepare attention mask
        extended_attention_mask = concat_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Encoder layers
        all_encoder_layers = []
        for layer_module in self.encoder:
            hidden_states = layer_module(hidden_states, extended_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        
        # Pooled output (use first token [CLS])
        pooled_output = self.pooler(hidden_states[:, 0])
        pooled_output = self.pooler_activation(pooled_output)
        
        return all_encoder_layers, pooled_output
