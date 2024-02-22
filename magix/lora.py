import re
from functools import partial
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
import flax


def adapt_params(params, lora_states, alpha=32):
    def adapt_one_param(p, l):
        if l is not None:
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
                assert len(param.shape) == 2, "LORA only works for 2D parameters"
                
                new_rng, in_rng = jax.random.split(prng, 2)
                in_dim, out_dim = param.shape
                in_mat = init(in_rng, (in_dim, lora_rank))
                out_mat = jnp.zeros((lora_rank, out_dim))
                return (in_mat, out_mat), new_rng
        
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