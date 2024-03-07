import re
import logging
from functools import partial
from typing import Any, Callable, Dict
from collections import namedtuple

import jax
import jax.numpy as jnp
import flax
from jax.sharding import PartitionSpec as PS, NamedSharding

from . import spmd_utils

logger = logging.getLogger(__name__)

LoraPair = namedtuple('LoraPair', ['in_matrix', 'out_matrix'])


def create_lora_sharding(sharding_config, mesh, lora_abs):
    def create_sharding_one(k, v):
        if v is None:  # skip 
            return None
        if isinstance(v, LoraPair):
            v = v[0]
            spec = spmd_utils.get_sharding(k, v, sharding_config)
            spec_in = PS(*spec[:-1], None)
            spec_out = PS(*spec[:-2], None, spec[-1])
            return LoraPair(NamedSharding(mesh, spec_in), NamedSharding(mesh, spec_out))
        else:
            return spmd_utils.get_sharding(k, v, sharding_config, mesh)      
    
    return jax.tree_util.tree_map_with_path(
        create_sharding_one, lora_abs, is_leaf=lambda x: isinstance(x, LoraPair))


def adapt_params(params, lora_states, alpha=32):
    def adapt_one_param(p, l):
        if l is not None:
            l = tuple(map(lambda x: jnp.astype(x, jnp.bfloat16), l))
            return p + (alpha / l[0].shape[1])*jnp.matmul(l[0], l[1])
        return p
    return jax.tree_map(adapt_one_param, params, lora_states)


def init_lora_params(prng, params, rules):
    # initialization guard
    assert rules is not None, "LORA rules must be provided for initialization"
    for v in rules.values():
        assert v > 0, "LORA rank must be greater than 0 for initialization"
    
    init = jax.nn.initializers.he_uniform()
    def init_one_param(prng, path, param):
        path_str = "/".join(path)
        for r in rules:
            if re.search(r, path_str):
                lora_rank = rules[r]
                assert len(param.shape) >= 2
                if len(param.shape) != 2:
                    logger.warn(
                        'Initializing LORA for a tensor parameter.'
                        'Will apply the decomposition to the last two dimensions.'
                    )
                
                new_rng, in_rng = jax.random.split(prng, 2)
                leading_dims = param.shape[:-2]
                in_dims = leading_dims + (param.shape[-2], lora_rank)
                out_dims = leading_dims + (lora_rank, param.shape[-1])
                in_mat = init(in_rng, in_dims)
                out_mat = jnp.zeros(out_dims)
                return LoraPair(in_mat, out_mat), new_rng
        
        return None, prng
    
    flat_params = flax.traverse_util.flatten_dict(params)
    lora_state = {}
    for path, param in flat_params.items():
        lora_matrices, prng = init_one_param(prng, path, param)
        lora_state[path] = lora_matrices
        
    return flax.traverse_util.unflatten_dict(lora_state)


class Lora:
    def __init__(
        self,
        alpha: float,
        rules: Dict[str, int]=None,
    ):
        self.apply = partial(adapt_params, alpha=alpha)
        self.init_params = partial(init_lora_params, rules=rules)