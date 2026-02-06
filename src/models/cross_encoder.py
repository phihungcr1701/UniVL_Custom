"""Cross-modal Encoder - Matching source UniVL structure for checkpoint compatibility"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import copy


class CrossEmbeddings(nn.Module):
    """Construct the embeddings from position and token_type embeddings."""
    def __init__(self, hidden_size: int, max_position_embeddings: int,
                 type_vocab_size: int, hidden_dropout_prob: float):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, concat_embeddings: torch.Tensor, concat_type: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_length = concat_embeddings.size(0), concat_embeddings.size(1)
        
        if concat_type is None:
            concat_type = torch.zeros(batch_size, seq_length, dtype=torch.long, device=concat_embeddings.device)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=concat_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        token_type_embeddings = self.token_type_embeddings(concat_type)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = concat_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CrossSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, attention_probs_dropout_prob: float):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

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

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class CrossSelfOutput(nn.Module):
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


class CrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int,
                 hidden_dropout_prob: float, attention_probs_dropout_prob: float):
        super().__init__()
        self.self = CrossSelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = CrossSelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class CrossIntermediate(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        return hidden_states


class CrossOutput(nn.Module):
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


class CrossLayer(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, intermediate_size: int,
                 hidden_dropout_prob: float, attention_probs_dropout_prob: float):
        super().__init__()
        self.attention = CrossAttention(hidden_size, num_attention_heads,
                                       hidden_dropout_prob, attention_probs_dropout_prob)
        self.intermediate = CrossIntermediate(hidden_size, intermediate_size)
        self.output = CrossOutput(hidden_size, intermediate_size, hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class CrossEncoder(nn.Module):
    def __init__(self, num_hidden_layers: int, hidden_size: int, num_attention_heads: int,
                 intermediate_size: int, hidden_dropout_prob: float, attention_probs_dropout_prob: float):
        super().__init__()
        layer = CrossLayer(hidden_size, num_attention_heads, intermediate_size,
                          hidden_dropout_prob, attention_probs_dropout_prob)
        # IMPORTANT: Must be named "layer" to match checkpoint: cross.encoder.layer.X
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor,
                output_all_encoded_layers: bool = True) -> list:
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class CrossPooler(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pool by taking the first token
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class CrossModel(nn.Module):
    """
    Cross-modal encoder model matching source UniVL structure
    Checkpoint keys: cross.embeddings.*, cross.encoder.layer.*, cross.pooler.*
    """
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
    ):
        super().__init__()
        self.embeddings = CrossEmbeddings(
            hidden_size, max_position_embeddings, type_vocab_size, hidden_dropout_prob
        )
        self.encoder = CrossEncoder(
            num_hidden_layers, hidden_size, num_attention_heads,
            intermediate_size, hidden_dropout_prob, attention_probs_dropout_prob
        )
        self.pooler = CrossPooler(hidden_size)

    def forward(
        self,
        concat_input: torch.Tensor,
        concat_type: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        output_all_encoded_layers: bool = True
    ) -> Tuple[list, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones(concat_input.size(0), concat_input.size(1), device=concat_input.device)
        if concat_type is None:
            concat_type = torch.zeros_like(attention_mask, dtype=torch.long)

        # Create 3D attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(concat_input, concat_type)
        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask, output_all_encoded_layers
        )
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        return encoded_layers, pooled_output
