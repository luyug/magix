
from typing import Optional
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import PartitionSpec as PS
from jax._src import mesh as mesh_lib

from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxCausalLMOutput,
)
from ..layer_transform import make_scan_stack
from .qwen2_model import Qwen2Config, FlaxQwen2RMSNorm
from . import qwen2_model as qwen2_model_orig


class FlaxQwen2ScanLayer(qwen2_model_orig.FlaxQwen2DecoderLayer):
    def __call__(self, *args, **kwargs):
        return super(FlaxQwen2ScanLayer, self).__call__(*args, **kwargs)[0]
    

class FlaxQwen2LayerCollection(nn.Module):
    config: Qwen2Config
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ):
        all_attentions = None  # () if output_attentions else None
        all_hidden_states = None  # () if output_hidden_states else None

        blocks = make_scan_stack(
            FlaxQwen2ScanLayer,
            self.config.num_hidden_layers,
            remat=True,
            deterministic=deterministic,
            init_cache=init_cache,
        ) (self.config, dtype=self.dtype, name='blocks')

        out, _ = blocks(
            (hidden_states, attention_mask, position_ids,)
        )
        
        hidden_states = out[0]
        
        # this contains possible `None` values - `FlaxQwen2Module` will filter them out
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs
    

class FlaxQwen2LMHeadWithNorm(nn.Module):
    config: Qwen2Config
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.norm = FlaxQwen2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype, name='norm')
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=jnp.bfloat16,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            name='lm_head',
        )
        
    def __call__(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        hidden_states = self.lm_head(hidden_states)
        return hidden_states
    
    
    
class FlaxQwen2ForCausalLMModule(nn.Module):
    config: Qwen2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.hidden_size = self.config.hidden_size
        embedding_init = jax.nn.initializers.normal(stddev=self.config.initializer_range)
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.hidden_size,
            embedding_init=embedding_init,
            dtype=jnp.bfloat16,
        )
        self.layers = FlaxQwen2LayerCollection(self.config, dtype=self.dtype)
        self.lm_head_with_norm = FlaxQwen2LMHeadWithNorm(self.config, dtype=self.dtype)
        
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ):
        input_embeds = self.embed_tokens(input_ids.astype("i4"))
        input_embeds = lax.with_sharding_constraint(input_embeds, PS('data', 'seq', 'model'))

        outputs = self.layers(
            input_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]

        hidden_states = lax.with_sharding_constraint(hidden_states, PS('data', 'seq', 'model'))
        logits = self.lm_head_with_norm(hidden_states)
        hidden_states = lax.with_sharding_constraint(hidden_states, PS('data', 'seq', 'model'))
    
        if not return_dict:
            return (logits,) + outputs[1:]
        return FlaxCausalLMOutput(
            logits=logits,
            hidden_states=outputs[1],
            attentions=outputs[2] if output_attentions else None,
        )
    
    
class FlaxQwen2ForCausalLM(qwen2_model_orig.FlaxQwen2PreTrainedModel):
    module_class = FlaxQwen2ForCausalLMModule
    
    partition_rules = {
        "embed_tokens/embedding": PS(("data", "seq"), "model"),
        "lm_head/kernel": PS(("data", "seq"), "model"),
        "lm_head/bias": PS("model"),
        "mlp/(gate|up)_proj": PS(None, ("data", "seq"), "model"),
        "mlp/down_proj": PS(None, "model", ("data", "seq")),
        "self_attn/(k|q|v)_proj/kernel": PS(None, ("data", "seq"), "model"),
        "self_attn/(k|q|v)_proj/bias": PS(None, "model"),
        "self_attn/o_proj": PS(None, "model", ("data", "seq")),
    }
    
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