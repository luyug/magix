import os
import logging

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from tqdm import tqdm, trange
from functools import partial

import jax
if jax.default_backend() == 'gpu':
    os.environ['XLA_FLAGS'] = (
    # '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=false '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_async_all_gather=true '
    '--xla_gpu_enable_async_reduce_scatter=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
    '--xla_gpu_collective_permute_decomposer_threshold=1024 '
    '--xla_gpu_all_reduce_combine_threshold_bytes=51200 '
    '--xla_gpu_simplify_all_fp_conversions=true '
)
import jax.numpy as jnp
import optax
import flax

from jax.sharding import Mesh
from jax.sharding import PartitionSpec as PS

import datasets
from transformers import AutoTokenizer
from simple_parsing import ArgumentParser
from simple_parsing.helpers import list_field


import magix
import magix.models
from magix import (
    get_chckpoint_manager,
    load_model_hub,
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
    
@dataclass
class ModelArgs:
    model_type: str = 'llama'
    model_name: str = None
    tokenizer_name: str = None
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
        0, train_args.learning_rate, int(total_train_steps*0.1), int(total_train_steps*0.9))

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
    mesh = magix.create_device_mesh(model_args.mesh_shape)
    
    checkpoint_manager = get_chckpoint_manager(train_args.checkpoint_dir, train_args.save_steps)
    is_new_train = checkpoint_manager.latest_step() is None
    
    _model_cls = magix.models.CAUSAL_LM_MODEL_MAPPING.get(model_args.model_type, None)
    if _model_cls is None:
        raise NotImplementedError(f"Model type {model_args.model_type} is not implemented")
    sharding_config = _model_cls.partition_rules
    
    if is_new_train:
        logger.info("Loading model from hub")
        model, params = load_model_hub(_model_cls, model_args.model_name, sharding_config, mesh, half=model_args.bf16_model_weights)
        opt_state = initialize_opt_state(optimizer, params, sharding_config, mesh)
    else:
        logger.info("Loading model from checkpoint")
        model, params, opt_state = load_model_and_optimizer_local(
            _model_cls, optimizer, checkpoint_manager, sharding_config, mesh, model_name=model_args.model_name)

    
    def train_step(params, opt_state, batch, dropout_rng):
        def compute_loss(params, batch):
            input_ids = batch['input_ids']
            attention_mask = jnp.logical_and(batch['attention_mask'][:,:-1], batch['attention_mask'][:,1:]).astype('bool')
            logits = model(
                input_ids=input_ids[:,:-1], attention_mask=attention_mask,
                params=params, train=True, dropout_rng=dropout_rng)[0]
            target_ids = input_ids[:,1:]
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
            loss = loss * attention_mask / attention_mask.sum()
            loss = loss.sum()
            return loss

        loss, grads = jax.value_and_grad(compute_loss, argnums=0) (params, batch)
        metrics = {"loss": loss}

        updates, new_opt_state = optimizer.update(grads, opt_state, params)  # transform & update state
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, metrics

    p_train_step = jax.jit(
        train_step,
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
    
    
    with mesh:
        for epoch in epochs:
            # Create sampling rng
            input_rng = jax.random.fold_in(data_rng, epoch)
            batch_loader = Batches(
                input_rng, train_dataset, train_args.batch_size, shuffle=True)
            steps_per_epoch = len(train_dataset) // train_args.batch_size
            # train
            for step in trange(steps_per_epoch):
                cur_step = epoch * (len(train_dataset) // train_args.batch_size) + step
                if lastest_step >= cur_step:
                    continue
                elif lastest_step == cur_step:
                    logger.info('Resuming training from step %d', cur_step)
                
                batch = batch_loader(step)
                dropout_rngs = jax.random.fold_in(dropout_rng, cur_step)
                params, opt_state, metrics = p_train_step(params, opt_state, batch, dropout_rngs)
                
                is_last_step = (cur_step + 1) == total_train_steps
                checkpoint_manager.save(
                    cur_step, items={'model': params, 'optimizer': opt_state}, force=is_last_step
                )
                train_metrics.append(metrics)
                
                if cur_step % 100 == 0 and cur_step > 0:
                    print(
                        f"Step... ({cur_step} | Loss: {combine_metrics(train_metrics)['loss'].mean()}, Learning Rate: {lr_schedule(cur_step)})",
                        flush=True,
                    )
                    train_metrics = []

            epochs.write(
                    f"Epoch... ({epoch + 1}/{train_args.num_epochs})"
                )

if __name__ == '__main__':
    main()