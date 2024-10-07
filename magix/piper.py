from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as PS

import flax.linen as nn

from . import layer_transform


def make_pipe_fwd(layer_cls, length_per_stage, **kwargs):
    scan_fwd_layer = layer_transform.make_scan_fwd_layer(layer_cls, **kwargs)
    layer_stack = nn.scan(
        scan_fwd_layer,
        length=length_per_stage,
        variable_axes={"params": 0},
        split_rngs={"params": True}
    )
    layer_pipe = nn.vmap(
        layer_stack,
        variable_axes={"params": 0},
        split_rngs={"params": True}
    )
    
    return layer_pipe


def make_pipe_bwd(layer_cls, **kwargs):
    scan_bwd_layer = layer_transform.make_scan_bwd_layer(layer_cls, **kwargs)
    layer_stack = nn.scan(
        scan_bwd_layer,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        reverse=True
    )
    layer_pipe = nn.vmap(
        layer_stack,
        variable_axes={"params": 0},
        split_rngs={"params": True})
    
    return layer_pipe


def create_pipe_buffer(
    pipe_fn,
    pre_fn,
    pipe_params,
    pre_params,
    x,
    n_stages,
    cycles=None,
):
    if cycles is not None:
        physical_stages = n_stages
        n_stages = n_stages * cycles
    
    
    single_stage_in_shape = jax.eval_shape(pre_fn, pre_params, x)
    
    a_buffer = jax.tree.map(
        lambda x: jnp.zeros((n_stages, *x.shape), dtype=x.dtype),
        single_stage_in_shape,
        is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
    )

    g_buffer = jnp.zeros_like(a_buffer[0])  # shape of the first term in inputs
    y_buffer = jnp.zeros_like(g_buffer[0])  # shape of the last stage input

    layer_in_shapes = jax.eval_shape(
        pipe_fn,
        pipe_params,
        a_buffer
    )[1]

    max_delays = 2*n_stages - 1
    l_buffer = jax.tree_map(
        lambda x: jnp.zeros((max_delays,) + x.shape, dtype=x.dtype),
        layer_in_shapes,
        is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
    )
    
    if cycles is not None:
        a_buffer = jax.tree_map(
            lambda x: x.reshape((cycles, physical_stages) + x.shape[1:]),
            a_buffer
        )
        l_buffer = jax.tree_map(
            lambda x: x.reshape((x.shape[0], cycles, physical_stages) + x.shape[2:]),
            l_buffer
        )
        g_buffer = g_buffer.reshape((cycles, physical_stages) + g_buffer.shape[1:])

    return a_buffer, l_buffer, g_buffer, y_buffer
    

def pad_inputs(
    inputs,
    label,
    n_stage: int,
    cycles: int = None,
):
    if cycles is not None:
        n_stage = n_stage * cycles
    
    input_right_padded = jax.tree.map(
        lambda x: jnp.concatenate(
            [x, jnp.zeros((2*n_stage - 1, *x.shape[1:]), dtype=x.dtype)], axis=0),
        inputs
    )
    input_left_padded = jax.tree_map(
        lambda x: jnp.concatenate(
            [jnp.zeros((2*n_stage - 1, *x.shape[1:]), dtype=x.dtype), x], axis=0),
        inputs
    )
    # label should be pad with n_stage on left and n_stage - 1 on right
    label_padded = jax.tree_map(
        lambda x: jnp.concatenate(
            [
                jnp.zeros((n_stage, *x.shape[1:]), dtype=x.dtype),
                x,
                jnp.zeros((n_stage - 1, *x.shape[1:]), dtype=x.dtype)
            ],
            axis=0
        ),
        label
    )
    
    return input_right_padded, input_left_padded, label_padded


@partial(jax.named_call, name='pipe_step')
def pipe_step(
    fwd_pipe,
    bwd_pipe,
    pre_fn,
    post_fn,
    x,
    tgt,
    x_delayed,
    pipe_state,
    params,
):
    a_buffer, l_buffer, g_buffer, y, step_idx, total_example_steps = pipe_state
    param_pre, param_pipe, param_post = params
    n_stage = g_buffer.shape[0]  # the gradient is supposed to be a single tensor

    run_fwd = step_idx - n_stage + 1 < total_example_steps
    run_bwd = step_idx >= n_stage
    bwd_eid = jnp.arange(-2*n_stage + 1, -n_stage + 1) + step_idx
    grad_valid = jnp.logical_and(bwd_eid >= 0, bwd_eid < total_example_steps)
    
    num_saved_activation_stage = jnp.arange(2*n_stage - 1, 0, -2)
    l_buffer_sl_indices = step_idx % num_saved_activation_stage
    

    ## fill first stage of input buffer ##########
    # a_buffer: Tuple[Array[stage, batch, ...]]
    with jax.named_scope('Pre-Pipe Function Fwd'):
        # def run_pre_fn_and_update_buffer():
        #     pre_out = pre_fn(param_pre, x)
        #     new_a_buffer = jax.tree_map(
        #         # lambda bf, out: bf.at[0].set(out),
        #         lambda bf, out: lax.dynamic_update_slice(
        #             bf, out[None, :], (0,) * bf.ndim),
        #         a_buffer,
        #         pre_out
        #     )
        #     return new_a_buffer
        
        # a_buffer = lax.cond(
        #     step_idx < total_example_steps,
        #     run_pre_fn_and_update_buffer,
        #     lambda: a_buffer,
        # )
        pre_fn_out = lax.cond(
            step_idx < total_example_steps,
            pre_fn,
            lambda *_: jax.tree.map(lambda x: jnp.zeros_like(x[0]), a_buffer),
            param_pre, x
        )
        a_buffer = jax.tree_map(
            lambda bf, out: bf.at[0].set(out),
            a_buffer,
            pre_fn_out
        )

    ## gather input from t_buffer ##########
    # l_buffer: Tuple[Array[max_delay, stage, batch, layer_per_stage, ..., feature]]
    saved = jax.tree_map(
        lambda bfs: jax.vmap(lambda i, bf: bf[i], in_axes=(0, 1), out_axes=0)(l_buffer_sl_indices, bfs),
        l_buffer
    )


    ## post_fn forward & backward ##########
    with jax.named_scope('Post-Pipe Function Fwd & Bwd'):
        def post_fn_fwd_bwd():
            loss, grads = jax.value_and_grad(post_fn, argnums=(0,1)) (param_post, y, tgt)
            post_params_grads, g = grads
            return loss, g, post_params_grads
        
        loss, g, post_params_grads = lax.cond(
            grad_valid[-1],
            post_fn_fwd_bwd,
            lambda: (0., jnp.zeros_like(y), jax.tree_map(lambda x: jnp.zeros_like(x), param_post)),
        )
        g = lax.with_sharding_constraint(g, PS('data', None, 'model'))
        g_buffer = g_buffer.at[-1].set(g)


    ## pipe_forward ##########
    def pipe_fwd():
        stage_out, layer_ins = fwd_pipe(param_pipe, a_buffer)
        return stage_out, layer_ins
    stage_out, layer_ins = lax.cond(
        run_fwd,
        pipe_fwd,
        lambda: (a_buffer, jax.tree_map(lambda x: jnp.zeros_like(x[0]), l_buffer)),
    )

    # y output
    new_y = stage_out[0]  # get rid of shared input
    new_y = new_y[-1] # last stage output
    
    # saved layer inputs for backward
    @partial(jax.vmap, in_axes=(1, 0, 0), out_axes=1)
    def update_l_buffer_one(b, x, i):
        return b.at[i].set(x)
    l_buffer = jax.tree_map(
        lambda bf, li: update_l_buffer_one(bf, li, l_buffer_sl_indices),
        l_buffer,
        layer_ins
    )
    
    # permute buffer in the fwd pipe
    a_buffer = jax.tree_map(
        lambda x: jnp.roll(x, 1, axis=0),
        stage_out
    )
    
    
    ## pipe_backward ##########
    def pipe_bwd():
        x_grads, pipe_param_grads = bwd_pipe(param_pipe, g_buffer, saved)

        g_pipe_out = x_grads[0]
        x_grads = jnp.roll(x_grads, -1, axis=0)

        pipe_param_grads = jax.tree_map(
            lambda x: jnp.where(grad_valid.reshape((-1,) + (1,) * (x.ndim - 1)), x, 0),
            pipe_param_grads
        )

        return g_pipe_out, pipe_param_grads, x_grads
    
    g_pipe_out, pipe_param_grads, g_buffer = lax.cond(
        run_bwd,
        pipe_bwd,
        lambda: (
            jnp.zeros_like(g_buffer[0]),
            jax.tree_map(lambda x: jnp.zeros_like(x), param_pipe),
            g_buffer
        ),
    )

    ## pre_fn backward ##########
    with jax.named_scope('Pre-Pipe Function Bwd'):
        def pre_fn_bwd():
            _, pre_bwk = jax.vjp(lambda *xx: pre_fn(*xx)[0], param_pre, x_delayed)
            pre_param_grads = pre_bwk(g_pipe_out)[0]
            return pre_param_grads
        pre_param_grads = lax.cond(
            grad_valid[0],
            pre_fn_bwd,
            lambda: jax.tree_map(lambda x: jnp.zeros_like(x), param_pre),
        )

    return loss, (a_buffer, l_buffer, g_buffer, new_y, step_idx + 1, total_example_steps), (pre_param_grads, pipe_param_grads, post_params_grads)