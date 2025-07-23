from __future__ import annotations

import math
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as PS
from jax.experimental.shard_map import shard_map
from jax._src import mesh as mesh_lib

from jax.experimental.pallas.ops.gpu import decode_attention
import numpy as np
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict

from transformers import Qwen2Config
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxCausalLMOutput,
)
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel
from transformers.utils import logging

from . import a as attention_module

logger = logging.get_logger(__name__)

try:
    from transformer_engine.jax import attention as te_attn
    _te_available = True
except ImportError:
    _te_available = False

# -----------------------------------------------------------------------------
# Rotary helpers (identical to Qwen2 / Qwen3)
# -----------------------------------------------------------------------------

def _create_sinusoidal_positions(num_pos: int, dim: int, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
    freqs = np.einsum("i,j->ij", np.arange(num_pos), inv_freq).astype("float32")
    embs = np.concatenate([np.sin(freqs), np.cos(freqs)], axis=-1)
    embs = embs.reshape(num_pos, 1, -1)
    return jnp.array(embs[:, :, :dim])

def _rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)

def _apply_rope(t, sin, cos):
    sin_full = jnp.concatenate([sin, sin], axis=-1)
    cos_full = jnp.concatenate([cos, cos], axis=-1)
    return (t * cos_full) + (_rotate_half(t) * sin_full)


# -----------------------------------------------------------------------------
# Normalisation layers – layer‑level RMSNorm only (no per‑head variant)
# -----------------------------------------------------------------------------

class FlaxQwen2RMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.weight = self.param(
            "weight", lambda _, s: jnp.ones(s, dtype=self.dtype), (self.hidden_size,)
        )

    def __call__(self, x: jax.Array):
        variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        x = x / jnp.sqrt(variance + self.eps)
        return (self.weight * x).astype(self.dtype)


# -----------------------------------------------------------------------------
# Rotary embedding module
# -----------------------------------------------------------------------------

class FlaxQwen2RotaryEmbedding(nn.Module):
    config: Qwen2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        hd = getattr(self.config, "head_dim", None) or (
            self.config.hidden_size // self.config.num_attention_heads
        )
        self.sincos = _create_sinusoidal_positions(
            self.config.max_position_embeddings, hd, base=self.config.rope_theta
        )
        # TODO: LongRoPE / Dynamic scaling hooks (see Qwen2.5 code)

    def __call__(self, k: jax.Array, q: jax.Array, pos_ids: jax.Array):
        sincos = self.sincos[pos_ids]
        sin, cos = jnp.split(sincos, 2, axis=-1)
        return (
            _apply_rope(k, sin, cos).astype(self.dtype),
            _apply_rope(q, sin, cos).astype(self.dtype),
        )


# -----------------------------------------------------------------------------
# Sliding‑window causal mask (helper)
# -----------------------------------------------------------------------------

def _sliding_window_mask(q_len: int, k_len: int, window: int, dtype="bool"):
    i = jnp.arange(q_len)[:, None]
    j = jnp.arange(k_len)[None, :]
    m = ((j <= i) & (j >= i - window + 1)).astype(dtype)
    return m[None, None, :, :]


# -----------------------------------------------------------------------------
# Attention – *no* per‑head RMSNorm, bias enabled, GQA + optional SWA
# -----------------------------------------------------------------------------

import jax.numpy as jnp

def count_prefix_zeros(x: jnp.ndarray) -> jnp.ndarray:
    """
    Return a 1‑D array whose i‑th element is the number of leading
    zeros in row i of `x`.

    Parameters
    ----------
    x : jnp.ndarray
        Any shape (..., n_cols). Works on arbitrary leading batch dims.

    Returns
    -------
    jnp.ndarray
        Same leading batch dims as `x`, with a single trailing length
        (one count per row).
    """
    zero_mask = (x == 0)
    prefix_mask = jnp.logical_and.accumulate(zero_mask, axis=-1)
    return prefix_mask.sum(axis=-1)


class FlaxQwen2Attention(nn.Module):
    config: Qwen2Config
    layer_idx: int
    dtype: jnp.dtype = jnp.float32
    fused_attention: bool = True  # Flash‑Attention stub
    causal: bool = True

    # ── helpers ────────────────────────────────────────────────────────────
    def _split_heads(self, x, n):
        return x.reshape(x.shape[:2] + (n, self.head_dim))

    def _merge_heads(self, x):
        return x.reshape(x.shape[:2] + (self.num_heads * self.head_dim,))

    def _repeat_kv(self, x):
        return jnp.repeat(x, self.group_factor, axis=-2)

    @nn.compact
    def _concatenate_to_cache(self, k, v, q, attn_mask):
        assert k.dtype == jnp.bfloat16
        assert v.dtype == jnp.bfloat16
        assert q.dtype == jnp.bfloat16

        is_init = self.has_variable("cache", "cached_key")
        ck = self.variable("cache", "cached_key", jnp.zeros, k.shape, k.dtype)
        cv = self.variable("cache", "cached_value", jnp.zeros, v.shape, v.dtype)
        idx = self.variable("cache", "cache_index", lambda: jnp.array(0, jnp.int32))

        if is_init:
            *b, max_len, _, _ = ck.value.shape
            cur = idx.value
            ck.value = lax.dynamic_update_slice(ck.value, k, (0,) * len(b) + (cur, 0, 0))
            cv.value = lax.dynamic_update_slice(cv.value, v, (0,) * len(b) + (cur, 0, 0))
            idx.value = cur + q.shape[1]

            pad_mask = jnp.broadcast_to(
                jnp.arange(max_len) < idx.value, tuple(b) + (1, q.shape[1], max_len)
            )
            attn_mask = combine_masks(pad_mask, attn_mask)
            k, v = ck.value, cv.value
        return k, v, attn_mask

    # ── setup ──────────────────────────────────────────────────────────────
    def setup(self):
        cfg = self.config
        self.head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.group_factor = self.num_heads // self.num_kv_heads

        dense = partial(
            nn.Dense,
            dtype=jnp.bfloat16,
            kernel_init=jax.nn.initializers.normal(cfg.initializer_range),
        )
        self.q_proj = dense(self.num_heads * self.head_dim, use_bias=True)
        self.k_proj = dense(self.num_kv_heads * self.head_dim, use_bias=True)
        self.v_proj = dense(self.num_kv_heads * self.head_dim, use_bias=True)
        self.o_proj = dense(cfg.hidden_size, use_bias=False)

        # 👉 NOTE: no per‑head RMSNorm here – omitted on purpose!
        
        self.rotary = FlaxQwen2RotaryEmbedding(cfg, dtype=jnp.float32)
        self.full_causal_mask = make_causal_mask(
            jnp.ones((1, cfg.max_position_embeddings), dtype="bool"), dtype="bool"
        )

    # ── forward ────────────────────────────────────────────────────────────
    def __call__(
        self,
        x: jax.Array,
        attn_mask: jax.Array,
        pos_ids: jax.Array,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        # linear projections
        def shard_cp(t):
            return lax.with_sharding_constraint(t, PS("data", 'seq', "model"))
        q = self._split_heads(shard_cp(self.q_proj(x)), self.num_heads)
        k = self._split_heads(shard_cp(self.k_proj(x)), self.num_kv_heads)
        v = self._split_heads(shard_cp(self.v_proj(x)), self.num_kv_heads)

        q = shard_cp(q)
        k = shard_cp(k)
        v = shard_cp(v)

        # rotary
        k, q = self.rotary(k, q, pos_ids)
        k = k.astype(jnp.bfloat16)
        q = q.astype(jnp.bfloat16)

        # masks (causal + optional padding)
        bs, q_len, k_len = x.shape[0], q.shape[1], k.shape[1]
        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal = lax.dynamic_slice(
                self.full_causal_mask, (0, 0, mask_shift, 0), (1, 1, q_len, max_decoder_length)
            )
        else:
            causal = self.full_causal_mask[:, :, :q_len, :k_len]
        causal = jnp.broadcast_to(causal, (bs,) + causal.shape[1:])


        if attn_mask.ndim == 2:
            attn_mask = attn_mask[:, None, None, :]
        attn_mask = jnp.broadcast_to(attn_mask, causal.shape)
        attn_mask = combine_masks(attn_mask, causal, dtype="bool")

        # PKV cache -----------------------------------------------------------
        if init_cache or self.has_variable("cache", "cached_key"):
            k, v, attn_mask = self._concatenate_to_cache(k, v, q, attn_mask)

        # choose path ---------------------------------------------------------
        use_fused = (
            self.fused_attention
            and _te_available
            # and not init_cache
            and not self.has_variable("cache", "cached_key")
            and q.shape[1] >= 32
        )
        
        is_decoding = self.has_variable("cache", "cached_key") and q.shape[1] == 1

        if use_fused:  # ---------- Flash / TE ----------
            q, k, v = map(lambda t: t.astype(jnp.bfloat16), (q, k, v))

            # if is_prefill:
            #     # we have left padding
            #     # only in the prefill stage will we trigger this
            #     num_pad_tokens = (input_mask == 0).sum(axis=-1)
            #     # seq_lens = q.shape[1] - num_pad_tokens
            #     seq_lens = jnp.full((q.shape[0],), 25, dtype=jnp.int32)  # TODO: use max length of the batch
                
            #     # jax.debug.print('seq_lens: {seq_lens}', seq_lens=seq_lens)
                
            #     mask_type = te_attn.AttnMaskType.PADDING_CAUSAL_BOTTOM_RIGHT_MASK
            # else:
            seq_lens = jnp.full((q.shape[0],), q.shape[1], dtype=jnp.int32)
            mask_type = te_attn.AttnMaskType.CAUSAL_MASK
            
            attn_out = te_attn.fused_attn(
                (q, k, v),
                None,
                te_attn.SequenceDescriptor.from_seqlens(seqlens=(seq_lens, seq_lens)),
                # te_attn.SequenceDescriptor.from_seqlens(seqlens=(input_mask[:,0,0].sum(1), input_mask[:,0,0].sum(1))),
                None,
                attn_bias_type=te_attn.AttnBiasType.NO_BIAS,
                attn_mask_type=mask_type,
                qkv_layout=te_attn.QKVLayout.BSHD_BSHD_BSHD,
                scaling_factor=1.0 / math.sqrt(self.head_dim),
                dropout_probability=0.0,
                is_training=not deterministic,
                context_parallel_axis='seq',
                context_parallel_strategy = te_attn.CPStrategy.DEFAULT,
            )
            attn_weights = None  # not returned by fused path
        # elif is_decoding:
        #     print('Decoding on mesh:', mesh_lib.thread_resources.env.physical_mesh, flush=True)
        #     @partial(
        #         shard_map,
        #         mesh = mesh_lib.thread_resources.env.physical_mesh,
        #         in_specs = (
        #             PS("data", "model"),
        #             PS("data", None, "model"),
        #             PS("data", None, "model"),
        #             PS('data'),
        #             PS('data'),
        #         ),
        #         check_rep=False,
        #         out_specs = PS("data", "model"),
        #     )
        #     def attend(q, k, v, start_idx, kv_seq_len):
        #         return attention_module.gqa(
        #             q, k, v, start_idx, kv_seq_len, block_h=8, num_stages=4, num_warps=8,
        #         )
            
        #     attn_out = attend(
        #         q.squeeze(1),
        #         k,
        #         v,
        #         count_prefix_zeros(attn_mask[:, 0, 0]).reshape((-1,)),
        #         jnp.full((q.shape[0],), self.variables["cache"]["cache_index"], dtype=jnp.int32).reshape((-1,)),
        #     )
        #     attn_out = attn_out[:, None, :, :]
        #     attn_weights = None
        else:             # ---------- vanilla ----------
            # group-query
            if self.num_kv_heads != self.num_heads:
                k, v = self._repeat_kv(k), self._repeat_kv(v)
            bias = jnp.where(
                attn_mask,
                jnp.zeros(attn_mask.shape, dtype=self.dtype),
                jnp.full(attn_mask.shape, jnp.finfo(self.dtype).min, dtype=self.dtype),
            )

            attn_weights = dot_product_attention_weights(q, k, bias=bias, dtype=jnp.float32, deterministic=deterministic)
            if self.dtype is not jnp.float32:
                attn_weights = attn_weights.astype(self.dtype)
            attn_out = jnp.einsum("...hqk,...khd->...qhd", attn_weights, v)

        # merge heads & output proj
        attn_out = self._merge_heads(attn_out)
        attn_out = self.o_proj(attn_out)

        return (attn_out, attn_weights) if output_attentions else (attn_out,)


# -----------------------------------------------------------------------------
# MLP with SwiGLU (same as Qwen2 / Qwen3)
# -----------------------------------------------------------------------------

class FlaxQwen2MLP(nn.Module):
    config: Qwen2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        hid = self.config.hidden_size
        inner = self.config.intermediate_size or hid * 4
        dense = partial(
            nn.Dense,
            use_bias=False,
            dtype=jnp.bfloat16,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.gate_proj = dense(inner)
        self.up_proj = dense(inner)
        self.down_proj = dense(hid)
        self.act = ACT2FN[self.config.hidden_act]

    def __call__(self, x):
        shard_cp = lambda t: lax.with_sharding_constraint(t, PS("data", 'seq', "model"))
        return self.down_proj(self.act(self.gate_proj(shard_cp(x))) * self.up_proj(shard_cp(x)))


# -----------------------------------------------------------------------------
# Decoder layer
# -----------------------------------------------------------------------------

class FlaxQwen2DecoderLayer(nn.Module):
    config: Qwen2Config
    dtype: jnp.dtype = jnp.float32
    layer_idx: int = 0

    def setup(self):
        self.input_layernorm = FlaxQwen2RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype
        )
        self.self_attn = FlaxQwen2Attention(self.config, self.layer_idx, dtype=jnp.bfloat16)
        self.post_attention_layernorm = FlaxQwen2RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype
        )
        self.mlp = FlaxQwen2MLP(self.config, dtype=jnp.bfloat16)

    @partial(
        nn.remat,
        static_argnums=(4, 5, 6),
    )
    def __call__(
        self,
        hs,
        attn_mask,
        pos_ids,
        deterministic=True,
        init_cache=False,
        output_attentions=False,
    ):
        res = hs
        hs = self.input_layernorm(hs)
        attn_out = self.self_attn(
            hs, attn_mask, pos_ids, deterministic, init_cache, output_attentions
        )
        hs = res + attn_out[0]
        res = hs.astype(jnp.bfloat16)
        hs = self.post_attention_layernorm(hs)
        hs = self.mlp(hs)
        hs = res + hs
        hs = hs.astype(jnp.bfloat16)
        return (hs,) + attn_out[1:]


# -----------------------------------------------------------------------------
# Layer stack
# -----------------------------------------------------------------------------

class FlaxQwen2LayerCollection(nn.Module):
    config: Qwen2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.blocks = [
            FlaxQwen2DecoderLayer(self.config, layer_idx=i, dtype=self.dtype, name=str(i))
            for i in range(self.config.num_hidden_layers)
        ]
        self.has_sliding = "sliding_attention" in self.config.layer_types

    def __call__(
        self,
        inputs_embeds,
        attn_mask,
        pos_ids,
        deterministic=True,
        init_cache=False,
        output_atts=False,
        output_hids=False,
    ):
        all_h, all_a = (), ()
        hs = inputs_embeds
        for blk in self.blocks:
            hs = lax.with_sharding_constraint(hs, PS("data", 'seq', "model"))
            out = blk(hs, attn_mask, pos_ids, deterministic, init_cache, output_atts)
            hs = out[0]
            if output_hids:
                all_h += (hs,)
            if output_atts:
                all_a += (out[1],)
        return hs, (all_h if output_hids else None), (all_a if output_atts else None)


# -----------------------------------------------------------------------------
# Main backbone
# -----------------------------------------------------------------------------

class FlaxQwen2Module(nn.Module):
    config: Qwen2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=jnp.bfloat16,
        )
        self.layers = FlaxQwen2LayerCollection(self.config, dtype=self.dtype)
        self.norm = FlaxQwen2RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic=True,
        init_cache=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        bs, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (bs, seq_len))
        embeds = self.embed_tokens(input_ids.astype("i4"))
        hs, all_h, all_a = self.layers(
            embeds,
            attention_mask,
            position_ids,
            deterministic,
            init_cache,
            output_attentions,
            output_hidden_states,
        )
        hs = self.norm(hs)
        if output_hidden_states:
            all_h += (hs,)
        return FlaxBaseModelOutput(last_hidden_state=hs, hidden_states=all_h, attentions=all_a)


# -----------------------------------------------------------------------------
# Pre‑trained wrappers & LM head
# -----------------------------------------------------------------------------

class FlaxQwen2PreTrainedModel(FlaxPreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    module_class: nn.Module = None

    partition_rules = {
        "embed_tokens/embedding": PS(("data", "seq"), "model"),
        "lm_head": PS(("data", "seq"), "model"),
        "mlp/(gate|up)_proj": PS(("data", "seq"), "model"),
        "mlp/down_proj": PS("model", ("data", "seq")),
        "self_attn/(k|q|v)_proj/kernel": PS(("data", "seq"), "model"),
        "self_attn/(k|q|v)_proj/bias": PS("model"),
        "self_attn/o_proj": PS("model", ("data", "seq")),
    }
    
    def __init__(self, config: Qwen2Config,  input_shape: Tuple = (1, 1), seed: int = 0, dtype=jnp.float32, _do_init=True, **kwargs):
        super().__init__(config, self.module_class(config=config, dtype=dtype, **kwargs), input_shape=input_shape,
                         seed=seed, dtype=dtype, _do_init=_do_init)
        
    def init_weights(self, rng, input_shape, params=None):
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attn_mask = jnp.ones_like(input_ids)
        pos_ids = jnp.broadcast_to(jnp.arange(input_shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        random_params = self.module.init({"params": params_rng, "dropout": dropout_rng}, input_ids, attn_mask, pos_ids, return_dict=False)["params"]
        if params is not None:
            flat_rand = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for k in self._missing_keys:
                params[k] = flat_rand[k]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        return random_params
    
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])
    
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxLlamaAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


class FlaxQwen2Model(FlaxQwen2PreTrainedModel):
    module_class = FlaxQwen2Module


# ────────────────────────────────────────────────────────────
# Causal‑LM head wrapper (weight‑tie with embeddings)
# ────────────────────────────────────────────────────────────

class FlaxQwen2ForCausalLMModule(nn.Module):
    config: Qwen2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.model = FlaxQwen2Module(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=jnp.bfloat16,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic=True,
        init_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        outputs = self.model(
            input_ids,
            attention_mask,
            position_ids,
            deterministic,
            init_cache,
            output_attentions,
            output_hidden_states,
        )
        logits = self.lm_head(lax.with_sharding_constraint(outputs.last_hidden_state, PS("data", "seq", "model")))
        if not return_dict:
            return (logits,) + (outputs.hidden_states, outputs.attentions)
        return FlaxCausalLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxQwen2ForCausalLM(FlaxQwen2PreTrainedModel):
    module_class = FlaxQwen2ForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        print('**** Input shape calling init_cache:', batch_size, max_length, flush=True)
        past_key_values = self.init_cache(batch_size, max_length)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since Llama uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


__all__ = [
    "FlaxQwen2Model",
    "FlaxQwen2ForCausalLM",
    "FlaxQwen2PreTrainedModel",
]
