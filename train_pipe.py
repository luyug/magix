import os

# os.environ.update({
#   "NCCL_LL128_BUFFSIZE": "-2",
#   "NCCL_LL_BUFFSIZE": "-2",
#    "NCCL_PROTO": "SIMPLE,LL,LL128",
#  })

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_cuda_data_dir=/sw/user/cudatoolkits/installs/cuda-12.2.0 "
    "--xla_gpu_graph_level=0 "
    "--xla_disable_hlo_passes=rematerialization "
    "--xla_gpu_shard_autotuning=false "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
    # "--xla_gpu_enable_triton_softmax_fusion=false "
    # "--xla_gpu_enable_triton_gemm=false "
    # "--xla_gpu_triton_gemm_any=false "
    # "--xla_gpu_enable_triton_hopper=false "
    # "--xla_gpu_all_reduce_combine_threshold_bytes=1073741824 "
    # "--xla_gpu_all_gather_combine_threshold_bytes=1073741824 "
    # "--xla_gpu_reduce_scatter_combine_threshold_bytes=134217728 "
    # "--xla_gpu_enable_pipelined_all_gather=true "
    # "--xla_gpu_enable_pipelined_reduce_scatter=true "
    # "--xla_gpu_enable_pipelined_all_reduce=true "
    # "--xla_gpu_enable_while_loop_double_buffering=true "
    "--xla_gpu_multi_streamed_windowed_einsum=true "
    "--xla_gpu_threshold_for_windowed_einsum_mib=4096 "
    "--xla_gpu_enable_all_gather_combine_by_dim=false "
    "--xla_gpu_enable_reduce_scatter_combine_by_dim=false "
    # "--xla_gpu_enable_custom_fusions=true "
    # "--xla_allow_excess_precision=true "
    # "--xla_gpu_enable_pipelined_p2p=true "  # <---- compile error
    "--xla_gpu_collective_permute_decomposer_threshold=1024 "
    "--xla_gpu_lhs_enable_gpu_async_tracker=true "
    "--xla_gpu_use_memcpy_local_p2p=true "
    "--xla_gpu_enable_nccl_user_buffers=false "
    "--xla_gpu_enable_nccl_comm_splitting=false "
    "--xla_gpu_enable_nccl_per_stream_comms=true "
    "--xla_gpu_enable_while_loop_reduce_scatter_code_motion=true " # <---- doesn't work
)

os.environ["NCCL_P2P_LEVEL"] = "NVL"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["NCCL_NVLS_ENABLE"] = "0"
os.environ["NCCL_IB_SL"] = "1"
os.environ["NCCL_NCHANNELS_PER_NET_PEER"] = "4"

os.environ["NCCL_NET_GDR_LEVEL"] = "PHB"
os.environ["NCCL_CROSS_NIC"] = "1"
os.environ["NCCL_COLLNET_ENABLE"] = "1"
os.environ["NCCL_NET"] = "AWS Libfabric"

os.environ["FI_CXI_DISABLE_HOST_REGISTER"] = "1"
os.environ["FI_MR_CACHE_MONITOR"] = "userfaultfd"
os.environ["FI_CXI_DEFAULT_CQ_SIZE"] = "131072"
os.environ["FI_CXI_RX_MATCH_MODE"] = "hybrid"
os.environ["FI_CXI_RDZV_PROTO"] = "alt_read"
os.environ["FI_CXI_REQ_BUF_SIZE"] = "8388608"


import jax
jax.distributed.initialize(
    local_device_ids=[0, 1, 2, 3],
)
if jax.process_index() == 0:
    print('Jax version:', jax.__version__, flush=True)
    print('Discovered devices:', jax.devices(), flush=True)


import logging

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from tqdm import tqdm, trange
from functools import partial


import jax.experimental
import jax.experimental.shard_map
import jax.numpy as jnp
import jax.experimental.mesh_utils
import jax._src.ad_checkpoint

from jax import lax
from jax.sharding import Mesh, PartitionSpec as PS

import optax
import flax
import numpy as np



import datasets
from transformers import AutoTokenizer, AutoConfig
from simple_parsing import ArgumentParser
from simple_parsing.helpers import list_field


import magix
import magix.models
import magix.piper
from magix import (
    get_chckpoint_manager,
    load_model_hub,
    load_model_local,
    load_model_and_optimizer_local,
    initialize_opt_state,
)

def apply_chat_template(turns: Iterable[Dict[str, str]], eos_token: str = None):
    ROLE_DICT = {
        'user': '<|user|>',
        'assistant': '<|assistant|>',
        'system': '<|system|>',
    }
    def _format(turn):
        role, content = turn['role'], turn['content']
        return f"{ROLE_DICT[role]}\n{content}{eos_token}"
    
    return '\n'.join(_format(turn) for turn in turns)


class TrainDataset:
    def __init__(
        self,
        train_data,
        tokenizer,
        field_name: str = 'text',
        max_len: int = 1024,
        use_chat_template: bool = False,
    ):
        self.data = train_data
        self.tokenizer = tokenizer
        self.field_name = field_name
        self.max_len = max_len
        self.use_chat_template = use_chat_template
        
    def __len__(self):
        return len(self.data)

    def get_batch(self, indices):
        batch = self.data[indices]
        batch = batch[self.field_name]
        if self.use_chat_template:
            batch = [apply_chat_template(turns, eos_token=self.tokenizer.eos_token) for turns in batch]
        tokenized = self.tokenizer(
            batch, max_length=self.max_len+1, padding='max_length',
            truncation=True, return_tensors='np',
        )
        return dict(tokenized)

class Batches:
    def __init__(
        self,
        rng: jax.random.PRNGKey,
        dataset: TrainDataset,
        batch_size: int,
        shuffle: bool = False
    ):
        steps_per_epoch = len(dataset) // batch_size

        if shuffle:
            batch_idx = jax.random.permutation(rng, len(dataset))
        else:
            batch_idx = jnp.arange(len(dataset))

        batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
        batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))
        
        self.dataset = dataset
        self.batch_idx = batch_idx
        
        self.host_mesh = jax.sharding.Mesh(
            np.array(jax.devices()).reshape(jax.process_count(), jax.local_device_count()), ['host', 'dev'])

        
    def __call__(self, step):
        idx = self.batch_idx[step]
        batch = self.dataset.get_batch(idx)
        return batch


def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: (path[-1] != "bias" and 'layernorm' not in path[-2]) for path in flat_params}
    return flax.traverse_util.unflatten_dict(flat_mask)


@dataclass
class TrainArgs:
    train_file: str = None
    train_data_config: str = None
    train_data_field: str = 'text'
    split: str = 'train'
    use_chat_template: bool = False
    checkpoint_dir: str = None
    max_length: int = 1024
    num_epochs: int = 1
    micro_batch_size: int = 1
    batch_size: int = 16
    num_target_passages: int = 16
    query_num_chunks: int = 4
    passage_num_chunks: int = 8
    learning_rate: float = 2e-6
    weight_decay: float = 0.0001
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    max_grad_norm: float = 1.0
    save_steps: int = 200
    seed: int = 42
    dup_lm_head: bool = False
    compute_loss_last_stage: bool = False
    profiling_steps: List[int] = list_field()
    
@dataclass
class ModelArgs:
    model_type: str = 'llama'
    model_name: str = None
    tokenizer_name: str = None
    model_cache_dir: str = None
    mesh_shape: List[int] = list_field(-1, 1)
    bf16_model_weights: bool = False
    config_name: str = None

OPT_SHARDING_RULE = {
    'embed_tokens/embedding': PS(('pipe', 'data'), 'model'),
    'lm_head/kernel': PS(('pipe', 'data'), 'model'),
    'mlp/(gate|up)_proj': PS('pipe', 'data', 'model'),
    'mlp/down_proj': PS('pipe', 'model', 'data'),
    'self_attn/(k|q|v)_proj': PS('pipe', 'data', 'model'),
    'self_attn/o_proj': PS('pipe', 'model', 'data'),
    'layernorm/weight': PS('pipe', ('data', 'model')),
}

COMPUTE_SHARDING_RULE = {
    'embed_tokens/embedding': PS('pipe', 'model'),
    'lm_head/kernel': PS(None, 'model'),
    'mlp/(gate|up)_proj': PS('pipe', None, 'model'),
    'mlp/down_proj': PS('pipe', 'model', None),
    'self_attn/(k|q|v)_proj': PS('pipe', None, 'model'),
    'self_attn/o_proj': PS('pipe', 'model', None),
    'layernorm/weight': PS('pipe'),
}

PIPE_GRAD_SHARDING_RULE = {
    'mlp/(gate|up)_proj': PS('pipe', None, 'model'),
    'mlp/down_proj': PS('pipe', None, 'model', None),
    'self_attn/(k|q|v)_proj': PS('pipe', None, None, 'model'),
    'self_attn/o_proj': PS('pipe', None, 'model', None),
    'layernorm/weight': PS('pipe'),
}

PIPE_ACCUM_SHARDING_RULE = {
    'mlp/(gate|up)_proj': PS('pipe', None, 'data', 'model'),
    'mlp/down_proj': PS('pipe', None, 'model', 'data'),
    'self_attn/(k|q|v)_proj': PS('pipe', None, 'data', 'model'),
    'self_attn/o_proj': PS('pipe', None, 'model', 'data'),
    'layernorm/weight': PS('pipe', None, ('data', 'model')),
}


def main():
    parser = ArgumentParser()
    parser.add_arguments(TrainArgs, dest="train_args")
    parser.add_arguments(ModelArgs, dest="model_args")
    args = parser.parse_args()
    train_args: TrainArgs = args.train_args
    model_args: ModelArgs = args.model_args
    
    # logger with date and time
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    # dataset setup
    if train_args.train_file.endswith('.jsonl'):
        train_data = datasets.load_dataset('json', data_files=train_args.train_file)['train']
    else:     
        train_data = datasets.load_dataset(
            train_args.train_file,
            train_args.train_data_config
        )[train_args.split]
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
        add_eos_token=not train_args.use_chat_template,
        use_fast=True, padding_side='right', legacy=False)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = TrainDataset(train_data, tokenizer, train_args.train_data_field, train_args.max_length, train_args.use_chat_template)
    
    # optimizer setup
    total_train_steps = len(train_dataset) // train_args.batch_size * train_args.num_epochs
    lr_schedule = optax.warmup_cosine_decay_schedule(
        0, train_args.learning_rate, int(total_train_steps*0.1), total_train_steps)

    optimizer = optax.adamw(
        lr_schedule,
        mask=decay_mask_fn,
        b1=train_args.adam_beta1,
        b2=train_args.adam_beta2,
        weight_decay=train_args.weight_decay,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(train_args.max_grad_norm),
        optimizer,
    )
    
    # initalize model parameters and optimizer state
    mesh = magix.create_device_mesh(model_args.mesh_shape, ('data', 'pipe', 'model'))
    n_stages = mesh.shape['pipe']
    n_replicas = mesh.shape['data']
    
    checkpoint_manager = get_chckpoint_manager(train_args.checkpoint_dir, train_args.save_steps)
    is_new_train = checkpoint_manager.latest_step() is None
    
    _model_cls = magix.models.CAUSAL_LM_MODEL_MAPPING.get(model_args.model_type, None)
    if _model_cls is None:
        raise NotImplementedError(f"Model type {model_args.model_type} is not implemented")
    
    if is_new_train:
        config = AutoConfig.from_pretrained(model_args.config_name)
        config.max_position_embeddings = train_args.max_length
        model, params = load_model_local(
            _model_cls, 
            model_args.model_name,
            OPT_SHARDING_RULE,
            mesh,
            model_config=config,
            half=model_args.bf16_model_weights
        )
        opt_state = initialize_opt_state(optimizer, params, OPT_SHARDING_RULE, mesh)
    else:
        model, params, opt_state = load_model_and_optimizer_local(
            _model_cls, optimizer, checkpoint_manager, OPT_SHARDING_RULE, mesh, model_name=model_args.model_name)

    import socket
    print(socket.gethostname(), "- Model loaded", flush=True)

    embed, fwd_pipe, bwd_pipe, lm_head = model.pipeline_functions(n_stages=n_stages)
    def train_step(params, opt_state, batch):
        input_ids = batch['input_ids']
            
        attention_mask = jnp.logical_and(
            batch['attention_mask'][:,:-1],
            batch['attention_mask'][:,1:]
        ).astype('bool')
        
        total_tokens = attention_mask.sum()
        
        def compute_grad(_params, input_ids, attention_mask):
            # create pipe inputs
            input_ids, attention_mask, labels = jax.tree.map(
                lambda x: jnp.reshape(x, (-1, train_args.micro_batch_size) + x.shape[1:]),
                (input_ids[:, :-1], attention_mask, input_ids[:, 1:])
            )
            total_example_steps = input_ids.shape[0]
            
            input_ids = jnp.array(input_ids, dtype=jnp.int32)
            labels = jnp.array(labels, dtype=jnp.int32)

            
            n_steps, mbsz, sequence_length = input_ids.shape
            position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, None, :], (n_steps, mbsz, sequence_length))
            
            
            inputs, inputs_delayed, labels = magix.piper.pad_inputs(
                (input_ids, attention_mask, position_ids),
                (labels, attention_mask),
                n_stages
            )
            
            def duplicate_lm_head(lm_head_params):
                def dup_one(x):
                    # layer norm params
                    if len(x.shape) == 1:
                        x = x[None, None, ...]
                        x = jnp.repeat(x, n_replicas, axis=0)
                        x = jnp.repeat(x, n_stages, axis=1)
                        x = lax.with_sharding_constraint(x, PS('data', 'pipe', 'model'))
                        return x
                    # kernel params
                    elif len(x.shape) == 2:
                        x = x.astype(jnp.bfloat16)
                        x = x[None, None, ...]
                        x = jnp.repeat(x, n_replicas, axis=0)
                        x = jnp.repeat(x, n_stages, axis=1)
                        x = lax.with_sharding_constraint(x, PS('data', 'pipe', None, 'model'))
                        return x
                    else:
                        raise ValueError(f"Unsupported shape {x.shape}")
                
                return jax.tree.map(dup_one, lm_head_params)
            
            def reduce_lm_head(lm_head_params):
                def reduce_one(x):
                    return jnp.sum(x, axis=(0, 1))
                
                return jax.tree.map(reduce_one, lm_head_params)
                
            def pre_fn(embed_params, xx):
                x, xs = xx[0], xx[1:]
                embeded = embed(embed_params, x)
                embeded = lax.with_sharding_constraint(embeded, PS('data', None, 'model'))
                return (embeded,) + xs
                
            def compute_loss(head_params, y, mini_batch_tgt):
                mini_batch_tgt, mini_batch_mask = mini_batch_tgt
                logits = lm_head(head_params, y)
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits.astype(jnp.float32),
                    mini_batch_tgt
                )
                
                loss = loss * mini_batch_mask / total_tokens
                loss = loss.sum()
                return loss
            
            def compute_loss_dup(loss_params, y, mini_batch_tgt):
                @partial(
                    jax.experimental.shard_map.shard_map,
                    mesh=mesh,
                    in_specs=(PS('data', 'pipe'), PS('data', 'pipe'), PS('data', 'pipe')),
                    out_specs=PS(),
                    auto=frozenset(['model'])
                )
                def _loss_fn(loss_params, y, mini_batch_tgt):
                    loss_params = jax.tree_map(jnp.squeeze, loss_params)
                    loss = compute_loss(loss_params, y, mini_batch_tgt)
                    return lax.psum(lax.psum(loss, 'pipe'), 'data')
                
                return _loss_fn(loss_params, y, mini_batch_tgt)
            
            all_params = model.reshape_to_pipeline_parameters(_params, n_stages)
            all_params = tuple({'params': p} for p in all_params)
            embed_params, pipe_params, head_params = all_params
            
            if train_args.dup_lm_head:
                head_params = duplicate_lm_head(head_params)
            
            all_params = embed_params, pipe_params, head_params

            if train_args.dup_lm_head:
                post_fn = compute_loss_dup
            else:
                post_fn = compute_loss
                    
            def build_buffer(pipe_params, embed_params, x):
                return magix.piper.create_pipe_buffer(
                    fwd_pipe, pre_fn,
                    pipe_params, embed_params,
                    x,
                    n_stages
                )
                    
            a_buffer, l_buffer, g_buffer, y_buffer = build_buffer(
                pipe_params, embed_params,
                jax.tree.map(lambda x: x[0], inputs)
            )

            a_buffer_sharding = (PS('pipe', 'data', None, 'model'),) + tuple(None for _ in range(len(a_buffer)-1))
            l_buffer_sharding = (PS(None, 'pipe', None, 'data', None, 'model'),) + tuple(PS(None, 'pipe', None, 'data') for _ in range(len(l_buffer)-1))
            g_buffer_sharding = PS('pipe', 'data', None, 'model')
            y_buffer_sharding = PS('data', 'pipe', 'model')
            
            buffer_sharding = (a_buffer_sharding, l_buffer_sharding, g_buffer_sharding, y_buffer_sharding)

            buffers = (a_buffer, l_buffer, g_buffer, y_buffer)
            if train_args.compute_loss_last_stage:
                buffers = lax.with_sharding_constraint(buffers, buffer_sharding)[:-1] + (y_buffer,)
            else:
                buffers = lax.with_sharding_constraint(buffers, buffer_sharding)
            pipe_state = buffers + (0, total_example_steps)        
            
            def pipe_scan_step(ic):
                i, carry = ic
                loss_acc, grads_acc, pipe_state = carry
                x, tgt, x_delayed = jax.tree.map(lambda x: x[i], (inputs, labels, inputs_delayed))
                
                loss, new_pipe_state, grads = magix.piper.pipe_step(
                    fwd_pipe, bwd_pipe, pre_fn, post_fn,
                    x, tgt, x_delayed, pipe_state, 
                    all_params
                )
                loss_acc += loss
                grads = jax.tree.map(lambda x: x.astype(jnp.bfloat16), grads)
                grads_acc = jax.tree.map(lambda acc, g: acc + g, grads_acc, grads)
                
                new_carry = (loss_acc, grads_acc, new_pipe_state)
                return (i+1, new_carry)

            # run the pipeline
            accum = jax.tree.map(lambda x: jnp.zeros(x.shape, jnp.bfloat16), all_params)
            accum = (
                accum[0],
                lax.with_sharding_constraint(accum[1], magix.spmd_utils.get_sharding_tree(accum[1], PIPE_ACCUM_SHARDING_RULE)),
                accum[2],
            )
            carry = (0.0, accum, pipe_state)
            n_micro_batches = jax.tree.leaves(inputs)[0].shape[0]
            idx_and_carry = (0, carry)
            
            _, carry_out = lax.while_loop(
                lambda x: x[0] < n_micro_batches,
                pipe_scan_step,
                idx_and_carry
            )
            loss, grads, _ = carry_out
            
            grads = [g['params'] for g in grads]
            if train_args.dup_lm_head:
                grads = grads[:2] + [reduce_lm_head(grads[2])]
            grads = model.reshape_from_pipeline_parameters(*grads)
            
            return loss, grads
        
        with jax.named_scope('ComputeGradient'):
            compute_params = model.to_bf16(params)
            compute_params = lax.with_sharding_constraint(compute_params, magix.spmd_utils.get_sharding_tree(params, COMPUTE_SHARDING_RULE))
            loss, grads = compute_grad(compute_params, input_ids, attention_mask)
            # grads = lax.optimization_barrier(grads)
            # grads = jax._src.ad_checkpoint._optimization_barrier(grads)
            
        metrics = {"loss": loss}
        with jax.named_scope('OptimizerStep'):
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, metrics



    p_train_step = jax.jit(
        train_step,
        donate_argnums=(0,1,2),
        out_shardings=(magix.item_sharding(params), magix.item_sharding(opt_state), None)
    )
    
    jax.experimental.multihost_utils.sync_global_devices('Model Load')
    
    
    rng = jax.random.key(train_args.seed)
    dropout_rng, data_rng = jax.random.split(rng)
    
    # train loop
    lastest_step = checkpoint_manager.latest_step()
    if lastest_step is None:
        lastest_step = -1
        
    train_metrics = []

    def combine_metrics(list_of_dicts):
        return {key: jnp.array([d[key] for d in list_of_dicts]) for key in list_of_dicts[0]}
    
    
    epochs = tqdm(range(train_args.num_epochs), desc=f"Epoch ... (1/{train_args.num_epochs})", position=0)
    
    logger.info("Starting training loop...")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", train_args.num_epochs)
    logger.info("  Instantaneous batch size = %d", train_args.batch_size)
    
    
    with mesh:
        for epoch in epochs:
            # Create sampling rng
            input_rng = jax.random.fold_in(data_rng, epoch)
            batch_loader = Batches(
                input_rng, train_dataset, train_args.batch_size, shuffle=True)
            steps_per_epoch = len(train_dataset) // train_args.batch_size
            # train
            _batch = batch_loader(0)
            for step in trange(steps_per_epoch, disable=jax.process_index() != 0):
                cur_step = epoch * (len(train_dataset) // train_args.batch_size) + step
                if lastest_step >= cur_step:
                    continue
                elif lastest_step == cur_step:
                    logger.info('Resuming training from step %d', cur_step)
                
                # batch = batch_loader(step)
                dropout_rngs = jax.random.fold_in(dropout_rng, cur_step)
                if step in train_args.profiling_steps:
                    from ctypes import cdll
                    libcudart = cdll.LoadLibrary('libcudart.so')
                    libcudart.cudaProfilerStart()
                    params, opt_state, metrics = p_train_step(params, opt_state, _batch)
                    jax.tree.map(lambda x: x.block_until_ready(), (params, opt_state))
                    libcudart.cudaProfilerStop()

                else:
                    params, opt_state, metrics = p_train_step(params, opt_state, _batch)
                
                is_last_step = (cur_step + 1) == total_train_steps
                # if cur_step > 0:
                #     checkpoint_manager.save(
                #         cur_step, items={'model': params, 'optimizer': opt_state}, force=is_last_step
                #     )
                train_metrics.append(jax.device_get(metrics))
                
                if cur_step % 10 == 0 and cur_step > 0:
                    combined_metrics = combine_metrics(train_metrics)
                    if jax.process_index() == 0:
                        print(
                            f"Step... ({cur_step} | Loss: {combined_metrics['loss'].mean()}, Learning Rate: {lr_schedule(cur_step)})",
                            flush=True,
                        )
                    train_metrics = []


            epochs.write(
                    f"Epoch... ({epoch + 1}/{train_args.num_epochs})"
                )

if __name__ == '__main__':
    main()