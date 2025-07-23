import os
import logging

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from tqdm import tqdm, trange
from functools import partial

# os.environ.update({
#   "NCCL_LL128_BUFFSIZE": "-2",
#   "NCCL_LL_BUFFSIZE": "-2",
#    "NCCL_PROTO": "SIMPLE,LL,LL128",
#  })

os.environ["XLA_FLAGS"] = (
    # "--xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_aarch64/23.11/cuda/12.3 "
    "--xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3 "
    # "--xla_gpu_graph_level=0 "
    "--xla_disable_hlo_passes=rematerialization "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
    # "--xla_gpu_shard_autotuning=false "
    # "--xla_gpu_enable_triton_softmax_fusion=true "
    # "--xla_gpu_enable_triton_gemm=false "
    # "--xla_gpu_triton_gemm_any=true "
    # "--xla_gpu_enable_triton_hopper=true "
    "--xla_gpu_all_reduce_combine_threshold_bytes=457179136 "
    "--xla_gpu_all_gather_combine_threshold_bytes=457179136 "
    "--xla_gpu_reduce_scatter_combine_threshold_bytes=8388608 "
    "--xla_gpu_enable_pipelined_all_gather=true "
    "--xla_gpu_enable_pipelined_reduce_scatter=true "
    "--xla_gpu_enable_pipelined_all_reduce=true "
    "--xla_gpu_enable_while_loop_double_buffering=true "
    "--xla_gpu_multi_streamed_windowed_einsum=true "
    "--xla_gpu_threshold_for_windowed_einsum_mib=4096 "
    # "--xla_gpu_use_memcpy_local_p2p=true "
    "--xla_gpu_enable_all_gather_combine_by_dim=false "
    "--xla_gpu_enable_reduce_scatter_combine_by_dim=false "
    # "--xla_gpu_enable_custom_fusions=true "
    # "--xla_allow_excess_precision=true "
    # "--xla_gpu_enable_while_loop_reduce_scatter_code_motion=true "
    # # "--xla_gpu_enable_nccl_user_buffers=true "
    # "--xla_gpu_enable_nccl_comm_splitting=true "
    # "--xla_gpu_enable_nccl_per_stream_comms=true "
)

# NCCL_P2P_LEVEL=NVL CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE=1 NCCL_IB_SL=1
os.environ["NCCL_P2P_LEVEL"] = "NVL"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["NCCL_NVLS_ENABLE"] = "1"
os.environ["NCCL_IB_SL"] = "1"
os.environ["NCCL_NCHANNELS_PER_NET_PEER"] = "4"

# # # export NCCL_NET_GDR_LEVEL=PHB
# # # export NCCL_CROSS_NIC=1
# # # export NCCL_COLLNET_ENABLE=1
# # # export NCCL_NET="AWS Libfabric"
# # # export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
# # # export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
# # # export FI_CXI_DISABLE_HOST_REGISTER=1
# # # export FI_MR_CACHE_MONITOR=userfaultfd
# # # export FI_CXI_DEFAULT_CQ_SIZE=131072
# # # export FI_CXI_RX_MATCH_MODE=software
# # # export FI_CXI_RDZV_PROTO=alt_read
# # # export FI_CXI_REQ_BUF_SIZE=8388608

# os.environ["NCCL_NET_GDR_LEVEL"] = "PHB"
# os.environ["NCCL_CROSS_NIC"] = "1"
# os.environ["NCCL_COLLNET_ENABLE"] = "1"
# # os.environ["NCCL_NET"] = "AWS Libfabric"
# # os.environ["NCCL_NET_PLUGIN"] = "aws"
# # os.environ["LD_LIBRARY_PATH"] = "/sw/user/nccl/aws-ofi.1.6.0/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
# # os.environ["LD_LIBRARY_PATH"] = "/soft/libraries/hwloc/lib/:" + os.environ.get("LD_LIBRARY_PATH", "")
# os.environ["FI_CXI_DISABLE_HOST_REGISTER"] = "1"
# os.environ["FI_MR_CACHE_MONITOR"] = "userfaultfd"
# os.environ["FI_CXI_DEFAULT_CQ_SIZE"] = "131072"
# os.environ["FI_CXI_RX_MATCH_MODE"] = "hybrid"
# os.environ["FI_CXI_RDZV_PROTO"] = "alt_read"
# os.environ["FI_CXI_REQ_BUF_SIZE"] = "8388608"


os.environ["NCCL_NET_GDR_LEVEL"] = "LOC"
os.environ["NCCL_CROSS_NIC"] = "1"
os.environ["NCCL_COLLNET_ENABLE"] = "1"
os.environ["NCCL_NET_GDR_READ"] = "1"
# os.environ["NCCL_NET"] = "AWS Libfabric"
# os.environ["NCCL_NET_PLUGIN"] = "aws"
# os.environ["LD_LIBRARY_PATH"] = "/sw/user/nccl/aws-ofi.1.6.0/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
# os.environ["LD_LIBRARY_PATH"] = "/soft/libraries/hwloc/lib/:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["FI_CXI_DISABLE_HOST_REGISTER"] = "1"
os.environ["FI_MR_CACHE_MONITOR"] = "userfaultfd"
os.environ["FI_CXI_DEFAULT_CQ_SIZE"] = "131072"
os.environ["FI_CXI_RX_MATCH_MODE"] = "software"
os.environ["FI_CXI_RDZV_PROTO"] = "alt_read"
os.environ["FI_CXI_REQ_BUF_SIZE"] = "8388608"

import jax
jax.distributed.initialize(
    # coordinator_address='gh071:5000',
    # num_processes=8,
    local_device_ids=[0, 1, 2, 3],
)

if jax.process_index() == 0:
    print(jax.devices(), flush=True)

import jax.numpy as jnp
import optax
import flax

import numpy as np

from jax.sharding import Mesh
from jax.sharding import PartitionSpec as PS

import datasets
from transformers import AutoTokenizer, AutoConfig
from simple_parsing import ArgumentParser
from simple_parsing.helpers import list_field

from torch.utils.data import DataLoader, IterableDataset

import magix
import magix.models
from magix import (
    get_chckpoint_manager,
    load_model_hub,
    load_model_local,
    load_model_and_optimizer_local,
    initialize_opt_state
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
    ):
        self.data = train_data
        self.tokenizer = tokenizer
        self.field_name = field_name
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)

    def get_batch(self, indices):
        batch = self.data[np.array(indices)]
        batch = batch[self.field_name]
        
        ct_fn = getattr(self.tokenizer, 'apply_chat_template', None)
        if ct_fn is None:
            logging.warning('apply_chat_template not found in tokenizer, using default function')
            ct_fn = apply_chat_template
        
        batch = [apply_chat_template(turns, eos_token=self.tokenizer.eos_token) for turns in batch]
        tokenized = self.tokenizer(
            batch, max_length=self.max_len+1, padding='max_length',
            truncation=True, return_tensors='np',
        )
        return dict(tokenized)
    
    def __getitem__(self, indices):
        return self.get_batch(indices)

class Batches(IterableDataset):
    def __init__(
        self,
        dataset,
        batch_size: int,
        start: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.total_step_per_epoch = len(dataset) // batch_size
        self.curr = start
        
    def __call__(self, step):
        return self._get_batch(step)
    
    def _get_batch(self, step):
        step = step % self.total_step_per_epoch
        indices = list(range(step*self.batch_size, (step+1)*self.batch_size))
        x = self.dataset[indices]
        x = {k: np.array(v) for k, v in x.items()}
        return x
    
    def __iter__(self):
        return self
    
    def __next__(self):
        batch = self._get_batch(self.curr)
        self.curr += 1
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
    accumulation_steps: int = 1
    
@dataclass
class ModelArgs:
    model_type: str = 'llama'
    model_name: str = None
    config_name: str = None
    tokenizer_name: str = None
    tokenizer_branch: str = None
    model_cache_dir: str = None
    mesh_shape: List[int] = list_field(-1, 1)
    bf16_model_weights: bool = False

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
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
        add_eos_token=not train_args.use_chat_template,
        revision=model_args.tokenizer_branch,
        use_fast=True, padding_side='right', legacy=False)
    
    if os.path.exists(train_args.train_file):
        train_dataset = datasets.load_dataset('json', data_files=train_args.train_file)['train']
    else:
        train_dataset = TrainDataset(
            datasets.load_dataset(train_args.train_file)[train_args.split],
            tokenizer=tokenizer,
            field_name=train_args.train_data_field,
            max_len=train_args.max_length
        )
    
    tokenizer.pad_token = tokenizer.eos_token
    
    
    # optimizer setup
    total_train_steps = len(train_dataset) // train_args.batch_size * train_args.num_epochs
    lr_schedule = optax.warmup_cosine_decay_schedule(
        0, train_args.learning_rate, int(total_train_steps*0.1), total_train_steps)
    # lr_schedule = optax.schedules.warmup_constant_schedule(
    #     train_args.learning_rate * 0.01, train_args.learning_rate, int(total_train_steps*0.1))

    optimizer = optax.adamw(
        lr_schedule,
        mask=decay_mask_fn,
        b1=train_args.adam_beta1,
        b2=train_args.adam_beta2,
        weight_decay=train_args.weight_decay,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(train_args.max_grad_norm),
        optimizer
    )
    optimizer = optax.apply_if_finite(optimizer, 10)
    
    # initalize model parameters and optimizer state
    mesh = magix.create_device_mesh(model_args.mesh_shape, names=('data', 'seq', 'model'))
    
    checkpoint_manager = get_chckpoint_manager(train_args.checkpoint_dir, train_args.save_steps)
    is_new_train = checkpoint_manager.latest_step() is None
    
    _model_cls = magix.models.CAUSAL_LM_MODEL_MAPPING.get(model_args.model_type, None)
    if _model_cls is None:
        raise NotImplementedError(f"Model type {model_args.model_type} is not implemented")
    sharding_config = _model_cls.partition_rules
    
    print(f"Sharing config: {sharding_config}", flush=True)
    
    if is_new_train:
        logger.info("Loading model from hub")
        config = AutoConfig.from_pretrained(model_args.config_name)
        config.max_position_embeddings = train_args.max_length
        model, params = load_model_local(
            _model_cls, 
            model_args.model_name,
            sharding_config,
            mesh,
            model_config=config,
            half=model_args.bf16_model_weights
        )
        opt_state = initialize_opt_state(optimizer, params, sharding_config, mesh)
    else:
        config = AutoConfig.from_pretrained(model_args.config_name)
        config.max_position_embeddings = train_args.max_length
        model, params, opt_state = load_model_and_optimizer_local(
            _model_cls, optimizer, checkpoint_manager, sharding_config, mesh, model_config=config)

    
    def train_step(params, opt_state, batch, dropout_rng):
        def compute_loss(params, batch):
            input_ids = batch['input_ids'][:, :-1]
            target_ids = batch['input_ids'][:, 1:]
            loss_mask = batch['attention_mask'][:, :-1]
            attention_mask = jnp.where(loss_mask == 0, jnp.array(False), jnp.array(True))
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask,
                params=params, train=True, dropout_rng=dropout_rng)[0]

            logits = logits.astype(jnp.float32)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
            loss = loss * attention_mask / attention_mask.sum()
            loss = loss.sum()
            return loss

        loss, grads = jax.value_and_grad(compute_loss, argnums=0) (params, batch)
        
        metrics = {"loss": loss}

        updates, new_opt_state = optimizer.update(grads, opt_state, params)  # transform & update state
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, metrics
    
    def train_step(params, opt_state, batch, dropout_rng):
        def compute_loss(params, batch):
            input_ids = batch['input_ids'][:, :-1]
            target_ids = batch['input_ids'][:, 1:]
            loss_mask = batch['attention_mask'][:, :-1]
            attention_mask = jnp.where(loss_mask == 0, jnp.array(False), jnp.array(True))

            logits = model(
                input_ids=input_ids, attention_mask=attention_mask,
                params=params, train=True, dropout_rng=dropout_rng)[0]

            logits = logits.astype(jnp.float32)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
            loss = loss * loss_mask / loss_mask.sum()
            loss = loss.sum()
            return loss

        loss, grads = jax.value_and_grad(compute_loss, argnums=0) (params, batch)
        
        metrics = {"loss": loss}

        updates, new_opt_state = optimizer.update(grads, opt_state, params)  # transform & update state
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, metrics

    def train_step_acc(params, opt_state, batch, dropout_rng, acc_steps=1):
        total_tokens = batch['loss_mask'].sum()
        
        def compute_loss(params, batch):
            input_ids = batch['input_ids'][:, :-1]
            target_ids = batch['input_ids'][:, 1:]
            loss_mask = batch['attention_mask'][:, :-1]
            dropout_rng = batch['random_rng']
            attention_mask = jnp.where(loss_mask == 0, jnp.array(False), jnp.array(True))
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask,
                params=params, train=True, dropout_rng=dropout_rng)[0]

            logits = logits.astype(jnp.float32)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
            loss = loss * attention_mask / total_tokens
            loss = loss.sum()
            return loss
        
        def scan_fn(carry, x):
            loss_acc, grad_acc = carry
            loss, grads = jax.value_and_grad(compute_loss, argnums=0) (params, x)
            
            loss_acc += loss
            grad_acc = jax.tree.map(jnp.add, grad_acc, grads)
            
            return (loss_acc, grad_acc), None
        
        batch = jax.tree.map(lambda x: x.reshape((acc_steps, -1) + x.shape[1:]), batch)

        batch['random_rng'] = jax.random.split(dropout_rng, acc_steps)
        (loss, grads), _ = jax.lax.scan(
            scan_fn,
            (0.0, jax.tree_map(partial(jnp.zeros_like, dtype=jnp.float32), params)),
            batch
        )
        
        metrics = {"loss": loss}
        updates, new_opt_state = optimizer.update(grads, opt_state, params)  # transform & update state
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, metrics

    p_train_step = jax.jit(
        train_step if train_args.accumulation_steps == 1 else partial(train_step_acc, acc_steps=train_args.accumulation_steps),
        donate_argnums=(0,1,2,3),
        out_shardings=(magix.item_sharding(params), magix.item_sharding(opt_state), None)
    )
    
    
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


    batch_loader = Batches(train_dataset, train_args.batch_size, start=lastest_step + 1)
    batch_loader = DataLoader(batch_loader, collate_fn=lambda x: x, batch_size=None, num_workers=1)
    batch_loader = iter(batch_loader)
    
    with mesh:
        for epoch in epochs:
            # Create sampling rng
            input_rng = jax.random.fold_in(data_rng, epoch)
            steps_per_epoch = len(train_dataset) // train_args.batch_size
            # train
            for step in trange(steps_per_epoch, disable=jax.process_index() != 0):
                cur_step = epoch * (len(train_dataset) // train_args.batch_size) + step
                if lastest_step >= cur_step:
                    continue
                elif lastest_step == cur_step:
                    logger.info('Resuming training from step %d', cur_step)
                
                # batch = batch_loader(step)
                batch = next(batch_loader)
                
                dropout_rngs = jax.random.fold_in(dropout_rng, cur_step)
                
                params, opt_state, metrics = p_train_step(params, opt_state, batch, dropout_rngs)
                
                is_last_step = (cur_step + 1) == total_train_steps
                checkpoint_manager.save(
                    cur_step, items={'model': params, 'optimizer': opt_state}, force=is_last_step
                )
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
    
    checkpoint_manager.wait_until_finished()

if __name__ == '__main__':
    main()