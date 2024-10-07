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

from .llama_model import LlamaConfig, FlaxLlamaRMSNorm
from . import llama_model as llama_model_orig

class FlaxLlamaScanLayer(llama_model_orig.FlaxLlamaDecoderLayer):
    def __call__(self, *args, **kwargs):
        return super(FlaxLlamaScanLayer, self).__call__(*args, **kwargs)[0]
    
class FlaxLlamaLayerCollection(nn.Module):
    config: LlamaConfig
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
            FlaxLlamaScanLayer,
            self.config.num_hidden_layers,
            remat=True,
            deterministic=deterministic,
            init_cache=init_cache,
        ) (self.config, dtype=self.dtype, name='blocks')

        out, _ = blocks(
            (hidden_states, attention_mask, position_ids,)
        )
        hidden_states = out[0]

        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs
    

class LlamaLMHeadWithNorm(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        hidden_states = FlaxLlamaRMSNorm(self.config, dtype=self.dtype, name="norm")(hidden_states)
        lm_logits = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            name="lm_head",
        ) (hidden_states)
        return lm_logits


class FlaxLlamaForCausalLMModule(nn.Module):
    config: LlamaConfig
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
        self.layers = FlaxLlamaLayerCollection(self.config, dtype=self.dtype)
        self.lm_head_with_norm = LlamaLMHeadWithNorm(self.config, dtype=self.dtype)

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
        input_embeds = lax.with_sharding_constraint(input_embeds, PS('data', None, 'model'))

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

        hidden_states = lax.with_sharding_constraint(hidden_states, PS('data', None, 'model'))
        lm_logits = self.lm_head_with_norm(hidden_states)
        hidden_states = lax.with_sharding_constraint(hidden_states, PS('data', None, 'model'))

        return (lm_logits,) + outputs[1:]
    
    
class FlaxLlamaForCausalLM(llama_model_orig.FlaxLlamaForCausalLM):
    module_class = FlaxLlamaForCausalLMModule
    
    partition_rules = {
        'embed_tokens/embedding': PS('data', 'model'),
        'lm_head/kernel': PS('data', 'model'),
        'mlp/(gate|up)_proj': PS(None, 'data', 'model'),
        'mlp/down_proj': PS(None, 'model', 'data'),
        'self_attn/(k|q|v)_proj': PS(None, 'data', 'model'),
        'self_attn/o_proj': PS(None, 'model', 'data'),
    }
    
    pipeline_partition_rules = {
        'embed_tokens/embedding': PS(('pipe', 'data'), 'model'),
        'lm_head/kernel': PS(('pipe', 'data'), 'model'),
        'mlp/(gate|up)_proj': PS('pipe', 'data', 'model'),
        'mlp/down_proj': PS('pipe', 'model', 'data'),
        'self_attn/(k|q|v)_proj': PS('pipe', 'data', 'model'),
        'self_attn/o_proj': PS('pipe', 'model', 'data'),
        'layernorm/weight': PS('pipe'),
    }
    
    def pipeline_functions(self, n_stages):
        from .. import piper
        config = self._config
        
        embed = nn.Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            dtype=jnp.bfloat16,
        ).apply
        
        lm_head = LlamaLMHeadWithNorm(config, dtype=jnp.bfloat16).apply

        fwd_layer = piper.make_pipe_fwd(
            FlaxLlamaScanLayer,
            config.num_hidden_layers // n_stages,
            init_cache=False,
            deterministic=False,
        ) (config, dtype=self.module.dtype, name='blocks').apply
        
        bwd_layer = piper.make_pipe_bwd(
            FlaxLlamaScanLayer,
            init_cache=False,
            deterministic=False,
        ) (config, dtype=self.module.dtype, name='blocks').apply
                
        return embed, fwd_layer, bwd_layer, lm_head


    def reshape_to_pipeline_parameters(self, params, n_stage):
        return (
            params['embed_tokens'],
            jax.tree_map(
                lambda x: jnp.reshape(x, (n_stage, -1) + x.shape[1:]), params['layers']['blocks']),
            params['lm_head_with_norm'],
        )
        
    def reshape_from_pipeline_parameters(
        self,
        embed_params,
        layer_params,
        lm_head_params,
    ):
        return {
            'embed_tokens': embed_params,
            'layers': {
                'blocks': jax.tree.map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), layer_params)
            },
            'lm_head_with_norm': lm_head_params,
        }