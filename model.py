from functools import lru_cache
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache

from config import ModelConfig


class Qwen3RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ):
        '''
        RMSNorm described in https://arxiv.org/abs/1910.07467
        
        Args:
            hidden_size: int, the dimension of the hidden states
            eps: float, the epsilon for the RMSNorm
        '''
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @torch.compile
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        act_fn: callable = None,
    ):
        '''
        A gated MLP with act_fn activation function (default: SwiGLU).
        
        Args:
            hidden_size: int, the dimension of the hidden states
            intermediate_size: int, the dimension of the intermediate states
            act_fn: callable, the activation function
        '''
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = act_fn if act_fn is not None else F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_positions: int,
        base: float = 10000.0,
        unsqueeze_dim: int = -2
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_positions = max_positions
        self.unsqueeze_dim = unsqueeze_dim
        
        # build cache with cos and sin values for each position
        inv_freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        positions = torch.arange(max_positions, dtype=torch.float)
        pos_freqs = einsum(positions, inv_freqs, "i,j -> i j")
        self.register_buffer("cos_sin_cache", torch.stack([pos_freqs.cos(), pos_freqs.sin()], dim=-1))
    
    def _apply_rotary_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x1, x2 = torch.chunk(x.float(), 2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat((y1, y2), dim=-1).to(x.dtype)

    @torch.compile
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.cos_sin_cache[positions].unbind(dim=-1)
        cos = cos.unsqueeze(dim=self.unsqueeze_dim)
        sin = sin.unsqueeze(dim=self.unsqueeze_dim)
        q = self._apply_rotary_emb(q, cos, sin)
        k = self._apply_rotary_emb(k, cos, sin)
        return q, k

@lru_cache(maxsize=1)
def get_rotary_embedding(dim: int, max_positions: int, base: float = 10000.0):
    return RotaryEmbedding(dim, max_positions, base)


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        n_kv_heads: int = None,
        head_dim: int = None,
        kv_head_dim: int = None,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        is_causal: bool = True,
        dropout_p: float = 0.0,
        max_positions: int = 8192,
    ):
        """
        Args:
            hidden_size: int, the dimension of the hidden states
            n_heads: int, the number of query heads
            n_kv_heads: int, the number of key and value heads (default: n_heads)
            head_dim: int, the dimension of the query heads (default: hidden_size // n_heads)
            kv_head_dim: int, the dimension of the key and value heads (default: hidden_size // n_kv_heads)
            rope_theta: float, the theta for the rope embedding
            rms_norm_eps: float, the epsilon for the RMSNorm
            is_causal: bool, whether the attention is causal
            dropout_p: float, the dropout probability
            max_positions: int, the maximum number of positions for the rotary embedding
        """
        super().__init__()
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"
        assert hidden_size % n_kv_heads == 0, "hidden_size must be divisible by n_kv_heads"
        
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // n_heads
        self.scale = self.head_dim ** -0.5
        self.is_causal = is_causal
        self.dropout_p = dropout_p

        self.rope_theta = rope_theta
        self.rotary_embedding = get_rotary_embedding(self.head_dim, max_positions, self.rope_theta)
        
        self.q_proj = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_size, bias=False)
        
        self.q_norm = Qwen3RMSNorm(self.head_dim, rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, rms_norm_eps)

    def batched_attention(
        self,
        x: torch.Tensor,
        dropout_p: float = 0.0,
        is_causal: bool = True
    ) -> torch.Tensor:
        '''
        Attention with batched sequences.
        
        Args:
            x: torch.Tensor, the input tensor of shape (batch_size, sequence_length, hidden_size)
            dropout_p: float, the dropout probability
            is_causal: bool, whether the attention is causal
        Returns:
            torch.Tensor, the output tensor of shape (batch_size, sequence_length, hidden_size)
        '''
        q = rearrange(self.q_proj(x), "b s (h d) -> b s h d", h=self.n_heads)
        k = rearrange(self.k_proj(x), "b s (h d) -> b s h d", h=self.n_kv_heads)
        v = rearrange(self.v_proj(x), "b s (h d) -> b s h d", h=self.n_kv_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rotary_embedding(q, k, torch.arange(x.size(1), device=x.device, dtype=torch.long))

        out = flash_attn_func(
            q, k, v,
            softmax_scale=self.scale,
            dropout_p=dropout_p,
            causal=is_causal
        ) # TODO: add kv cache support

        out = rearrange(out, "b s h d -> b s (h d)")

        out = self.o_proj(out)

        return out

    def packed_attention(
        self,
        x: torch.Tensor,
        positions: torch.Tensor = None,
        cu_seqlens_q: torch.Tensor = None,
        cu_seqlens_k: torch.Tensor = None,
        max_seqlen_q: int = None,
        max_seqlen_k: int = None,
        dropout_p: float = 0.0,
        is_causal: bool = True,
    ) -> torch.Tensor:
        '''
        Attention with packed sequences. i.e. ['ABC', 'DEF', 'GHI'] -> ['ABCDEFGHI']
        
        Args:
            x: torch.Tensor, packed input tensor of shape (packed_sequence_length, hidden_size)
            positions: torch.Tensor, the start/endpositions of the sequences in the input tensor
            cu_seqlens_q: torch.Tensor, the cumulative sequence lengths of sequences in the query tensor
            cu_seqlens_k: torch.Tensor, the cumulative sequence lengths of sequences in the key tensor
            max_seqlen_q: int, the maximum sequence length of the query tensor
            max_seqlen_k: int, the maximum sequence length of the key tensor
            dropout_p: float, the dropout probability
            is_causal: bool, whether the attention is causal
        Returns:
            torch.Tensor, the output tensor of shape (packed_sequence_length, hidden_size)
        '''
        if positions is None or cu_seqlens_q is None or cu_seqlens_k is None or max_seqlen_q is None or max_seqlen_k is None:
            raise ValueError("positions, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, and max_seqlen_k must be provided for packed sequences")
        
        q = rearrange(self.q_proj(x), "s (h d) -> s h d", h=self.n_heads)
        k = rearrange(self.k_proj(x), "s (h d) -> s h d", h=self.n_kv_heads)
        v = rearrange(self.v_proj(x), "s (h d) -> s h d", h=self.n_kv_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rotary_embedding(q, k, positions)

        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            softmax_scale=self.scale,
            dropout_p=dropout_p,
            causal=is_causal
        ) # TODO: add kv cache support

        out = rearrange(out, "s h d -> s (h d)")
        out = self.o_proj(out)

        return out

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor=None,
        cu_seqlens_q: torch.Tensor=None,
        cu_seqlens_k: torch.Tensor=None,
        max_seqlen_q: int=None,
        max_seqlen_k: int=None,
        dropout_p: float = 0.0,
        is_causal: bool = True,
    ) -> torch.Tensor:
        '''
        Args:
            x: torch.Tensor, input tensor of shape:
                (batch_size, sequence_length, hidden_size) for batched sequences
                (packed_sequence_length, hidden_size) for packed sequences
            positions: torch.Tensor, start/end positions of the sequences in the input tensor or None for batched sequences
            cu_seqlens_q: torch.Tensor, cumulative sequence lengths of sequences in the query tensor or None for batched sequences
            cu_seqlens_k: torch.Tensor, cumulative sequence lengths of sequences in the key tensor or None for batched sequences
            max_seqlen_q: int, maximum sequence length of the query tensor or None for batched sequences
            max_seqlen_k: int, maximum sequence length of the key tensor or None for batched sequences
            dropout_p: float, dropout probability
            is_causal: bool, whether the attention is causal or not
        Returns:
            torch.Tensor, output tensor of shape:
                (batch_size, sequence_length, hidden_size) for batched sequences
                (packed_sequence_length, hidden_size) for packed sequences
        '''
        if x.ndim == 3:
            return self.batched_attention(x, dropout_p, is_causal)
        elif x.ndim == 2:
            return self.packed_attention(x, positions, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, is_causal)
        else:
            raise ValueError("x must be a tensor of shape (batch_size, sequence_length, hidden_size) for batched sequences or (packed_sequence_length, hidden_size) for packed sequences")


class Qwen3DecoderLayer(nn.Module):
    def __init__(
            self,
        hidden_size: int,
        intermediate_size: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rope_theta: float,
        rms_norm_eps: float,
        is_causal: bool,
        dropout_p: float,
        max_positions: int,
        act_fn: callable = None,
    ):
        """
        Args:
            hidden_size: int, the dimension of the hidden states
            intermediate_size: int, the dimension of the intermediate MLPstates
            n_heads: int, the number of query heads
            n_kv_heads: int, the number of key and value heads (default: n_heads)
            rope_theta: float, the theta for the rope embedding
            rms_norm_eps: float, the epsilon for the RMSNorm
            is_causal: bool, whether the attention is causal or not
            dropout_p: float, dropout probability
            max_positions: int, the maximum number of positions for the rotary embedding
            act_fn: callable, the activation function (default: SwiGLU)
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            act_fn=act_fn
        )
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            is_causal=is_causal,
            dropout_p=dropout_p,
            max_positions=max_positions
        )
        self.input_layernorm = Qwen3RMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden_size, rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor = None,
        cu_seqlens_q: torch.Tensor=None,
        cu_seqlens_k: torch.Tensor=None,
        max_seqlen_q: int=None,
        max_seqlen_k: int=None,
        dropout_p: float = 0.0,
        is_causal: bool = True,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            positions=positions,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            is_causal=is_causal
        )
        hidden_states = hidden_states + residual
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        
        return hidden_states

class Qwen3Model(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                n_heads=config.num_attention_heads,
                head_dim=config.head_dim,
                n_kv_heads=config.num_key_value_heads,
                rope_theta=config.rope_theta,
                rms_norm_eps=config.rms_norm_eps,
                is_causal=config.is_causal,
                dropout_p=config.attention_dropout,
                max_positions=config.max_position_embeddings,
                act_fn=F.silu
            )
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor = None,
        cu_seqlens_q: torch.Tensor = None,
        cu_seqlens_k: torch.Tensor = None,
        max_seqlen_q: int = None,
        max_seqlen_k: int = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                positions=positions,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=self.config.attention_dropout,
                is_causal=self.config.is_causal
            )
        hidden_states = self.norm(hidden_states)
        
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor = None,
        cu_seqlens_q: torch.Tensor = None,
        cu_seqlens_k: torch.Tensor = None,
        max_seqlen_q: int = None,
        max_seqlen_k: int = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k
        )
        logits = self.lm_head(hidden_states)
        
        return logits