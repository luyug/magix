"""
Single controller reinforcement learning in JAX.
"""
import os
import logging
import json

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import jax.experimental
import jax.experimental.multihost_utils
from tqdm import tqdm, trange
from functools import partial

import magix.score
import magix.score.math_verify

# os.environ.update({
#   "NCCL_LL128_BUFFSIZE": "-2",
#   "NCCL_LL_BUFFSIZE": "-2",
#    "NCCL_PROTO": "SIMPLE,LL,LL128",
#  })

qwen2_chat_template = """{%- for message in messages %}
  {%- if message.role == 'system' %}
<|im_start|>system
{{ message.content }}<|im_end|>
  {%- elif message.role == 'user' %}
<|im_start|>user
{{ message.content }}<|im_end|>
  {%- elif message.role == 'assistant' %}
{% generation %}<|im_start|>assistant
{{ message.content }}<|im_end|>
{% endgeneration %}
  {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{% generation %}<|im_start|>assistant
{% endgeneration %}
{%- endif %}"""

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
    # "--xla_gpu_enable_pipelined_all_gather=true "
    # "--xla_gpu_enable_pipelined_reduce_scatter=true "
    # "--xla_gpu_enable_pipelined_all_reduce=true "
    # "--xla_gpu_enable_while_loop_double_buffering=true "
    # "--xla_gpu_multi_streamed_windowed_einsum=true "
    # "--xla_gpu_threshold_for_windowed_einsum_mib=4096 "
    # "--xla_gpu_use_memcpy_local_p2p=true "
    "--xla_gpu_enable_all_gather_combine_by_dim=false "
    "--xla_gpu_enable_reduce_scatter_combine_by_dim=false "
    # "--xla_gpu_enable_custom_fusions=true "
    # "--xla_allow_excess_precision=true "
    # "--xla_gpu_enable_while_loop_reduce_scatter_code_motion=true "
    # "--xla_gpu_enable_nccl_user_buffers=true "
    "--xla_gpu_enable_nccl_comm_splitting=true "
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

from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as PS
from jax._src.sharding_impls import TransferToMemoryKind

from jax.experimental import mesh_utils
from jax.experimental import multihost_utils

import datasets
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer
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
import magix.spmd_utils as spmd_utils

import magix.score.math as math_scoring


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

def verify_answers(responses: List[str], raw_batch: Dict[str, Any], repeat: int) -> List[float]:
    if 'solution' in raw_batch:  # math
        solution = raw_batch['solution']
        gt_answers = [math_scoring.remove_boxed(math_scoring.last_boxed_only_string(sol)) for sol in solution]
    elif 'answer' in raw_batch:  # gsm
        gt_answers = raw_batch['answer']
        gt_answers = [g.split('####')[-1] for g in gt_answers]
    elif 'reward_model' in raw_batch:  # dapo
        gt_answers = raw_batch['reward_model']
        gt_answers = [x['ground_truth'] for x in gt_answers]
    else:
        raise ValueError("Ground truth answers not found in the batch. Please provide 'solution' or 'reward_model' field.")
    gt_answers = [[gt] * repeat for gt in gt_answers]
    gt_answers = [item for sublist in gt_answers for item in sublist]
    
    assert len(responses) == len(gt_answers), "Responses and ground truth answers must have the same length"

    scores = [magix.score.math_verify.compute_score(resp, gt) for resp, gt in zip(responses, gt_answers)]
    return np.array(scores, dtype=jnp.float32)


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
        
        _apply_chat_template = getattr(self.tokenizer, 'apply_chat_template', None)
        if _apply_chat_template is None:
            logging.warning('apply_chat_template not found in tokenizer, using default function')
            _apply_chat_template = apply_chat_template
        else:
            _apply_chat_template = partial(
                _apply_chat_template, tokenize=False, enable_thinking=False,
                add_generation_prompt=True,
            )
        
        def apply_ctemplate(turns):
            assert isinstance(turns, list), "Input should be a list of turns"
            return _apply_chat_template(turns, eos_token=self.tokenizer.eos_token)
        
        self.apply_chat_template = apply_ctemplate
        
    def __len__(self):
        return len(self.data)

    def get_batch(self, indices):
        raw_batch = self.data[np.array(indices)]
        batch = raw_batch[self.field_name]

        if isinstance(batch[0], str):
            batch = [[{'role': 'user', 'content': str(turn)}] for turn in batch]
        
        raw_prompt = batch

        formatted_batch = [self.apply_chat_template(turns) for turns in batch]

        tokenized = self.tokenizer(
            formatted_batch, max_length=self.max_len,
            truncation=True, return_tensors='np',
            padding=True,
            pad_to_multiple_of=256,
        )
        return dict(tokenized), raw_prompt, raw_batch
    
    def __getitem__(self, indices):
        return self.get_batch(indices)

class Batches(IterableDataset):
    def __init__(
        self,
        dataset,
        batch_size: int,
        start: int = 0,
        rng: Optional[jnp.ndarray] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.total_step_per_epoch = len(dataset) // batch_size
        self.curr = start
        self.rng = rng
        self.epoch = 0
        self.indices = None
        
        # Use CPU context manager to ensure arrays are allocated on CPU
        with jax.default_device(jax.local_devices(backend="cpu")[0]):
            self._create_epoch_indices()
        
    def _create_epoch_indices(self):
        if self.rng is not None:
            indices = jnp.arange(len(self.dataset))
            indices = jax.random.permutation(jax.random.fold_in(self.rng, self.epoch), indices)
            self.indices = np.array(indices)
        else:
            self.indices = np.arange(len(self.dataset))
        
    def __call__(self, step):
        return self._get_batch(step)
    
    def _get_batch(self, step):
        step = step % self.total_step_per_epoch
        current_epoch = step // self.total_step_per_epoch
        
        # If we've moved to a new epoch, reshuffle the indices
        if current_epoch > self.epoch and self.rng is not None:
            self.epoch = current_epoch
            with jax.default_device(jax.devices("cpu")[0]):
                self._create_epoch_indices()
            
        start_idx = step * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_indices = self.indices[start_idx:end_idx]  # type: ignore
        
        x, raw_prompt, raw_batch = self.dataset[batch_indices]
        x = {k: jnp.array(v) for k, v in x.items()}
        return x, raw_prompt, raw_batch

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
    warmup_steps: int = 128
    weight_decay: float = 0.0001
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    max_grad_norm: float = 1.0
    save_steps: int = 10
    seed: int = 42
    accumulation_steps: int = 1
    train_mb_size: int = 16
    log_prob_mb_size: int = 64
    group_size: int = 4
    algorithm_variant: str = 'dapo-clip'
    no_update_ntoks: int = 1024
    
@dataclass
class ModelArgs:
    model_type: str = 'llama'
    model_name: Optional[str] = None
    config_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    tokenizer_branch: Optional[str] = None
    model_cache_dir: Optional[str] = None
    mesh_shape: List[int] = list_field(-1, 1)
    gen_mesh_shape: List[int] = list_field(1, 1, 4)
    bf16_model_weights: bool = False


@dataclass
class InferenceArgs:
    generation_temperature: float = 1.0
    generation_top_p: float = 0.7
    generation_prompt_max_length: int = 128
    generation_max_length: int = 1024
    generation_mb_size: int = 16
    penalization_margin: int = 256
    

def params_to_local_mesh_for_inference(
    params: Dict[str, Any], 
    global_mesh: Mesh, 
    local_mesh: Mesh, 
    inf_pspes: Any, 
):
    @partial(
        jax.jit,
        out_shardings=jax.tree.map(lambda ps: NamedSharding(global_mesh, ps), inf_pspes)
    )
    def to_bf16(pp):
        def to_bf16_one(p):
            if p.ndim > 1:
                return p.astype(jnp.bfloat16)
            else:  # for vectors, we keep them in float32
                return p
        return jax.tree_map(lambda x: to_bf16_one(x), pp)
    params = to_bf16(params)

    params = multihost_utils.global_array_to_host_local_array(params, global_mesh, inf_pspes)
    params = multihost_utils.host_local_array_to_global_array(params, local_mesh, inf_pspes)

    return params

        

def compute_penalization(
    responses: jnp.ndarray,
    eos_token_id: int,
    max_possible_tokens: int,
    penalization_margin: int,
) -> jnp.ndarray:
    """Compute penalization for responses that exceed the allowed token limit.
    
    Args:
        responses: Generated token sequences
        tokenizer: Tokenizer with eos_token_id
        generation_max_length: Maximum generation length
        generation_prompt_max_length: Maximum prompt length
        penalization_margin: Margin before penalization starts
        
    Returns:
        Penalization values for each response
    """
    n_generated_tokens = (responses != eos_token_id).sum(axis=-1)

    no_penality_max_tokens = max_possible_tokens - penalization_margin
    overflow_n_tokens = jnp.maximum(0, n_generated_tokens - no_penality_max_tokens)
    penalization = overflow_n_tokens / penalization_margin
    return penalization


def main():
    parser = ArgumentParser()
    parser.add_arguments(TrainArgs, dest="train_args")
    parser.add_arguments(ModelArgs, dest="model_args")
    parser.add_arguments(InferenceArgs, dest="inference_args")
    args = parser.parse_args()
    train_args: TrainArgs = args.train_args
    model_args: ModelArgs = args.model_args
    inference_args: InferenceArgs = args.inference_args
    
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
        add_eots_token=True,
        add_eos_token=True,
        revision=model_args.tokenizer_branch,
        use_fast=True, padding_side='left', legacy=False)
    train_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
        add_eots_token=True,
        add_eos_token=True,
        revision=model_args.tokenizer_branch,
        use_fast=True, padding_side='right', legacy=False
    )

    
    if os.path.exists(train_args.train_file):
        train_dataset = datasets.load_dataset('json', data_files=train_args.train_file)['train']
    else:
        ds = datasets.load_dataset(
            train_args.train_file,
            train_args.train_data_config,
        )[train_args.split]
        ds = ds.filter(
            lambda x: len(tokenizer(x[train_args.train_data_field][0]['content'])['input_ids']) <= inference_args.generation_prompt_max_length,
            num_proc=16,
            keep_in_memory=True,
        )
        train_dataset = TrainDataset(
            ds,
            tokenizer=tokenizer,
            field_name=train_args.train_data_field,
            max_len=inference_args.generation_prompt_max_length,
        )

    # ------------------------------------------------------------
    # Setup optimization
    # ------------------------------------------------------------
    
    # Setup learning rate schedule
    total_train_steps = len(train_dataset) // train_args.batch_size * train_args.num_epochs
    lr_schedule_warmup = optax.linear_schedule(
        init_value=0.0,
        end_value= train_args.learning_rate,
        transition_steps=train_args.warmup_steps,
    )
    lr_schedule_constant = optax.constant_schedule(value=train_args.learning_rate)
    lr_schedule = optax.join_schedules(
        schedules=[lr_schedule_warmup, lr_schedule_constant],
        boundaries=[train_args.warmup_steps],
    )
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
    
    # ------------------------------------------------------------
    # Setup global mesh for training and local mesh for generation
    # ------------------------------------------------------------
    mesh = magix.create_device_mesh(model_args.mesh_shape, names=('data', 'seq', 'model'))    


    # Assume one generator per host, and each host has multiple devices
    local_mesh = Mesh(
        devices=mesh_utils.create_device_mesh(
            model_args.gen_mesh_shape,
            jax.local_devices(),
        ),
        axis_names=('data', 'seq', 'model'),
    )
    generation_global_mesh = Mesh(
        devices=mesh_utils.create_device_mesh(
            (jax.process_count(),) + tuple(model_args.gen_mesh_shape),
            jax.devices(),
        ),
        axis_names=('rep', 'data', 'seq', 'model'),
    )
    n_generation_replicas = jax.device_count() // jax.local_device_count()
    per_generation_replica_prompts = train_args.batch_size // n_generation_replicas
    data_start_idx = per_generation_replica_prompts * jax.process_index()
    data_end_idx = data_start_idx + per_generation_replica_prompts

    #### End of local topo info. ####
    
    checkpoint_manager = get_chckpoint_manager(
        train_args.checkpoint_dir,
        save_steps=train_args.save_steps,
        max_to_keep=6,
    )
    is_new_train = checkpoint_manager.latest_step() is None
    
    _model_cls = magix.models.CAUSAL_LM_MODEL_MAPPING.get(model_args.model_type, None)
    if _model_cls is None:
        raise NotImplementedError(f"Model type {model_args.model_type} is not implemented")
    sharding_config = _model_cls.partition_rules
    
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

    # bookkeep sharding info
    params_sharding = magix.item_sharding(params)
    opt_sharding = magix.item_sharding(opt_state)
    inf_sharding = spmd_utils.get_sharding_tree(
        params,
        spmd_utils.duplicate_over(sharding_config, ('data', 'seq'))
    )
     


    def train_step(
        params: Dict[str, Any],
        opt_state: Dict[str, Any],
        batch: Dict[str, Any],
        dropout_rng: jnp.ndarray,
        log_prob_mb_size: int = 64,
        train_mb_size: int = 16,
        group_size: int = 4,
        variant: str = 'dapo',
    ):
        print('Using algorithm variant: ', variant, flush=True)
        # ------------------------------------------------------------
        # Compute behave log probs
        # ------------------------------------------------------------
        def compute_behave_log_probs(
            behave_params: Dict[str, Any],
            input_ids: jnp.ndarray, 
            generated_ids: jnp.ndarray, 
            attention_mask: jnp.ndarray, 
        ):
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask,
                params=behave_params, train=False)[0]
            logprobs = jax.nn.log_softmax(logits, axis=-1)
            generated_token_logprobs  = jnp.take_along_axis(logprobs, generated_ids[..., None], axis=-1).squeeze(-1)
            generated_token_logprobs = generated_token_logprobs.astype(jnp.float32)
            return generated_token_logprobs
        
        # ------------------------------------------------------------
        # Scan_fn for compute_behave_log_probs
        # ------------------------------------------------------------
        def log_prob_scan_fn(params, x):
            input_ids = x['input_ids'][:, :-1]
            generated_ids = x['input_ids'][:, 1:]
            attention_mask = x['attention_mask'][:, :-1]
            generated_token_logprobs = compute_behave_log_probs(
                params,
                input_ids,
                generated_ids,
                attention_mask,
            )
            return params, generated_token_logprobs
        
        # ------------------------------------------------------------
        # Compute RL loss
        # A simple implementation of DAPO loss
        # ------------------------------------------------------------
        def compute_rl_loss(
            params,
            input_ids: jnp.ndarray,
            generated_ids: jnp.ndarray, 
            attention_mask: jnp.ndarray, 
            behave_log_probs: jnp.ndarray, 
            advantages: jnp.ndarray,
            dropout_rng: jnp.ndarray,
            assistant_masks: jnp.ndarray,
            variant: str,
            no_update_ntoks: int,
        ):
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask,
                params=params, train=True, dropout_rng=dropout_rng)[0]
            logprobs = jax.nn.log_softmax(logits, axis=-1)
            generated_token_logprobs  = jnp.take_along_axis(logprobs, generated_ids[..., None], axis=-1).squeeze(-1)
            generated_token_logprobs = generated_token_logprobs.astype(jnp.float32)
            
            ratio = jnp.exp(generated_token_logprobs - jax.lax.stop_gradient(behave_log_probs))
            if variant == 'dapo':
                surr1 = jnp.clip(ratio, 1 - 0.2, 1 + 0.25) * advantages[:, None]
                surr2 = ratio * advantages[:, None]
                objective = jnp.minimum(surr1, surr2)
            elif variant == 'dapo-clip':
                objective = jnp.clip(ratio, 1 - 0.2, 1 + 0.2) * advantages[:, None]
            elif variant == 'dapo-cap':
                # do clipping for positive advantages
                # do update cap for negative advantages
                ratio_no_grad = jax.lax.stop_gradient(ratio)
                clipped_ratio = jnp.clip(ratio_no_grad, 1 - 0.2, 1 + 0.2)
                loss_clip_scale = ratio_no_grad / clipped_ratio
                objective = ratio * loss_clip_scale * advantages[:, None]
            elif variant == "dapo-softclip":
                clipped_ratio = 1 + 0.2 * jnp.tanh((ratio - 1) / 0.2 * 1.0)
                objective = clipped_ratio * advantages[:, None]
            else:
                raise ValueError(f"Invalid variant: {variant}")
            
            loss_mask = assistant_masks * attention_mask
            
            # check for over-confidence
            # consider p=0.99
            # get it in log space
            OVER_CONFIDENCE_THRESHOLD = jnp.log(0.99)
            over_confidence_toks = behave_log_probs > OVER_CONFIDENCE_THRESHOLD
            
            def has_long_span(a: jnp.ndarray, L: int) -> jnp.ndarray:
                x  = a.astype(jnp.int32)
                cs = jnp.cumsum(x, axis=1)                          # (B, W)
                pad = jnp.pad(cs, ((0, 0), (L, 0)))                 # left‑pad zeros
                window_sum = cs - pad[:, :-L]                       # (B, W)

                return (window_sum >= L).any(axis=1)

            removed_objective = has_long_span(over_confidence_toks, no_update_ntoks)
            objective = jnp.where(
                removed_objective[:, None],
                jax.lax.stop_gradient(objective),
                objective
            )
            loss = -jnp.sum(objective * loss_mask) / jnp.sum(loss_mask)
            return loss, {'num_removed': removed_objective.sum()}
        
        # ------------------------------------------------------------
        # Scan_fn for compute_rl_loss
        # ------------------------------------------------------------
        def train_scan_fn(p_and_opt_state, x):
            params, opt_state = p_and_opt_state
            input_ids = x['input_ids'][:, :-1]
            generated_ids = x['input_ids'][:, 1:]
            attention_mask = x['attention_mask'][:, :-1]
            generated_token_logprobs = x['behave_log_probs']
            advantages = x['advantages']
            dropout_rng = x['random_rng']
            assistant_masks = x['assistant_masks'][:, :-1]
            if train_args.accumulation_steps > 1:
                micro_batches = {
                    'input_ids': input_ids,
                    'generated_ids': generated_ids,
                    'attention_mask': attention_mask,
                    'generated_token_logprobs': generated_token_logprobs,
                    'advantages': advantages,
                    'assistant_masks': assistant_masks,
                }
                micro_batches = jax.tree.map(lambda x: x.reshape((train_args.accumulation_steps, -1) + x.shape[1:]), micro_batches)
                micro_batches['random_rng'] = jax.random.split(dropout_rng, train_args.accumulation_steps)
                grad_acc = jax.tree.map(lambda x: jnp.zeros_like(x), params)
                compute_loss = partial(compute_rl_loss, variant=variant, no_update_ntoks=train_args.no_update_ntoks)
                
                def grad_acc_fn(g_acc, mb):
                    (loss, metrics), grads = jax.value_and_grad(compute_loss, argnums=0, has_aux=True) (
                        params,
                        mb['input_ids'],
                        mb['generated_ids'],
                        mb['attention_mask'],
                        mb['generated_token_logprobs'],
                        mb['advantages'],
                        mb['random_rng'],
                        mb['assistant_masks'],
                    )
                    g_acc = jax.tree.map(lambda grad, acc: acc + grad / train_args.accumulation_steps, grads, g_acc)
                    return g_acc, {'loss': loss, **metrics}
                
                grad_acc, metrics = jax.lax.scan(
                    grad_acc_fn,
                    grad_acc,
                    micro_batches
                )
                grads = grad_acc
                metrics['loss'] = metrics['loss'].mean()
                metrics['num_removed'] = metrics['num_removed'].sum()

            else:
                (loss, metrics), grads = jax.value_and_grad(compute_rl_loss, argnums=0, has_aux=True) (
                    params,
                    input_ids,
                    generated_ids,
                    attention_mask,
                    generated_token_logprobs,
                    advantages,
                    dropout_rng,
                    assistant_masks,
                    variant,
                    no_update_ntoks=train_args.no_update_ntoks,
                )
                metrics = {'loss': loss, **metrics}
            
            updates, new_opt_state = optimizer.update(grads, opt_state, params)  # transform & update state
            new_params = optax.apply_updates(params, updates)
            
            return (new_params, new_opt_state), metrics
        
        rewards = batch.pop('rewards')
        
        # compute behave log probs
        log_prob_batches = jax.tree.map(lambda x: x.reshape((-1, log_prob_mb_size) + x.shape[1:]), batch)
        _, behave_log_probs = jax.lax.scan(
            log_prob_scan_fn,
            params,
            log_prob_batches
        )
        behave_log_probs = behave_log_probs.reshape((-1,) + behave_log_probs.shape[2:])
        behave_log_probs = jax.lax.stop_gradient(behave_log_probs)
        
        batch['behave_log_probs'] = behave_log_probs
        
        # compute advantages
        grouped_rewards = jnp.reshape(rewards, (train_args.batch_size, train_args.group_size))
        mean_rewards = grouped_rewards.mean(axis=-1, keepdims=True)
        std_rewards = grouped_rewards.std(axis=-1, keepdims=True)
        
        
        assert std_rewards.shape == (train_args.batch_size, 1)
        assert mean_rewards.shape == (train_args.batch_size, 1)
        advantages = (grouped_rewards - mean_rewards) / (std_rewards + 1e-8)
        advantages = advantages.reshape((-1,))
        batch['advantages'] = advantages
        
        
        # sort batch by sign of variance
        sorted_indices = jnp.argsort(std_rewards.squeeze(-1) > 0, descending=True)
        # group the batch first
        batch = jax.tree.map(lambda x: x.reshape((train_args.batch_size, train_args.group_size) + x.shape[1:]), batch)
        batch = jax.tree.map(lambda x: jnp.take(x, sorted_indices, axis=0), batch)
        # flatten the batch
        batch = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), batch)
        
        # create micro-batches
        train_num_mb = train_args.batch_size * train_args.group_size // train_mb_size
        batch = jax.tree.map(lambda x: x.reshape((-1, train_mb_size) + x.shape[1:]), batch)
        batch['random_rng'] = jax.random.split(dropout_rng, train_num_mb)
        
        non_zero_advantages = batch['advantages'] != 0
        non_zero_advantages = jnp.reshape(
            non_zero_advantages,
            (train_num_mb, train_args.train_mb_size // train_args.group_size, train_args.group_size)
        )
        full_update_mini_batches = non_zero_advantages.any(axis=2).all(axis=1)
        n_updates = full_update_mini_batches.sum()
        
        
        # train
        def train_for_fn(i, p_o_m, mini_batches):
            params, opt_state, metrics = p_o_m
            mini_batch = jax.tree.map(lambda x: x[i], mini_batches)
            (params, opt_state), batch_metrics = train_scan_fn((params, opt_state), mini_batch)
            
            metrics = jax.tree.map(lambda x, y: x + y, metrics, batch_metrics)
            return params, opt_state, metrics
        
        train_for_fn = partial(train_for_fn, mini_batches=batch)
        
        def _train_fn(p_and_opt_state, batch):
            all_advantages_nonzero = jnp.all(batch['advantages'] != 0)
            
            def _train_fn_positive(p_and_opt_state, batch):
                p_and_opt_state, metrics = train_scan_fn(p_and_opt_state, batch)
                return p_and_opt_state, {**jax.tree.map(lambda x: x.mean(), metrics), 'skip_batch': 0}
            
            def _train_fn_negative(p_and_opt_state, batch):
                return p_and_opt_state, {'loss': 0.0, 'num_removed': 0., 'skip_batch': 1}
            
            return jax.lax.cond(
                all_advantages_nonzero,
                _train_fn_positive,
                _train_fn_negative,
                p_and_opt_state,
                batch
            )
            
            # if positive_advantages.sum() > 0:
            #     return train_scan_fn(p_and_opt_state, batch)
            # else:
            #     return p_and_opt_state, {'loss': 0.0, 'num_removed': 0}
        
        # (new_params, new_opt_state), metrics = jax.lax.scan(
        #     train_scan_fn,
        #     (params, opt_state),
        #     batch
        # )
        new_params, new_opt_state, metrics = jax.lax.fori_loop(
            0, n_updates, train_for_fn, (params, opt_state, {'loss': 0.0, 'num_removed': 0})
        )
        
        
        
        metrics = {
            "loss": metrics['loss'] / n_updates,
            "num_removed": metrics['num_removed'] / n_updates,
            "avg_reward": rewards.mean() / n_updates,
            "avg_advantage": advantages.mean() / n_updates,
            "n_updates": n_updates,
            # "skip_batch": metrics['skip_batch'].sum(),
        }
        
        return new_params, new_opt_state, metrics
    # ------------------------------------------------------------
    # (p)JIT train_step
    # ------------------------------------------------------------
    p_train_step = jax.jit(
        partial(
            train_step, 
            train_mb_size=train_args.train_mb_size, 
            log_prob_mb_size=train_args.log_prob_mb_size,
            group_size=train_args.group_size,
            variant=train_args.algorithm_variant,
        ),
        donate_argnums=(0,1,2,3),
        out_shardings=(params_sharding, opt_sharding, None)  # type: ignore
    )
    
    # ------------------------------------------------------------
    # (p)JIT generate
    # ------------------------------------------------------------
    @partial(
        jax.jit,
        static_argnames=('sample', 'temperature', 'max_length', 'top_p'),  # TODO: investigate max_length tracing error in transformer.generate
        out_shardings=NamedSharding(local_mesh, PS()),  # type: ignore
        donate_argnums=(3,)
    )
    def generate(
        params: Dict[str, Any],
        inputs: jnp.ndarray,
        mask: jnp.ndarray,
        rng_key: jnp.ndarray,
        sample=False,
        top_p=0.7,
        temperature=1.0,
        max_length: int = 1024,
    ):
        generation = model.generate(
            inputs,
            attention_mask=mask,
            prng_key=rng_key,
            max_length=max_length,
            params=params,
            do_sample=sample,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
        ).sequences
        
        return generation
    
    
    init_rng = jax.random.key(train_args.seed)
    dropout_rng, data_rng, inf_rng = jax.random.split(init_rng, 3)
    
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


    batch_loader = Batches(train_dataset, train_args.batch_size, start=lastest_step + 1, rng=data_rng)
    # batch_loader = DataLoader(batch_loader, collate_fn=lambda x: x, batch_size=None, num_workers=0)
    batch_loader = iter(batch_loader)
    
    for epoch in epochs:
        steps_per_epoch = len(train_dataset) // train_args.batch_size
        
        # transient states for generation
        cpu_opt_state = None
        inference_params = None
        
        for step in trange(steps_per_epoch, disable=jax.process_index() != 0):
            # Always advance the batch loader
            batch, raw_prompt, raw_batch = next(batch_loader)
            cur_step = epoch * (len(train_dataset) // train_args.batch_size) + step
            if lastest_step >= cur_step:
                continue
            elif lastest_step == cur_step:
                logger.info('Resuming training from step %d', cur_step)    

            ## Generation
            # offload opt_state to pinned host memory
            cpu_opt_state = jax.jit(jax.device_put, donate_argnums=0, static_argnums=(1,)) (
                opt_state,
                TransferToMemoryKind('unpinned_host')            
            )
            jax.tree.map(lambda x: x.delete(), opt_state)
            # end cpu offload
            
            # Reshard: fp32 global params -> bf16 local params
            inference_params = params_to_local_mesh_for_inference(
                params,
                generation_global_mesh,
                local_mesh,
                inf_sharding
            )
            
            # grab local batch via index_slice
            batch = jax.tree.map(
                lambda x: x[data_start_idx:data_end_idx],
                batch
            )
            # repeat batch for rl group size
            batch = jax.tree.map(
                lambda x: jnp.repeat(x, train_args.group_size, axis=0),
                batch
            )
            # create micro-batches
            batch = jax.tree.map(
                lambda x: jnp.reshape(x, (-1, inference_args.generation_mb_size) + x.shape[1:]),
                batch
            )

            n_mb = batch['input_ids'].shape[0]
            input_len = batch['input_ids'].shape[-1]
            responses = []
            inf_rng_this_step, inf_rng = jax.random.split(inf_rng)
            
            with local_mesh:
                for mb_idx in trange(n_mb, desc=f"Rank {jax.process_index()} Generation"):
                    batch_this_mb = jax.tree.map(
                        lambda x: x[mb_idx],
                        batch
                    )
                    rng, inf_rng_this_step = jax.random.split(inf_rng_this_step)
                    batch_this_mb['random_rng'] = rng
                    generation = generate(
                        inference_params,
                        batch_this_mb['input_ids'],
                        batch_this_mb['attention_mask'],
                        batch_this_mb['random_rng'],
                        sample=True,
                        top_p=inference_args.generation_top_p,
                        temperature=inference_args.generation_temperature,
                        max_length=inference_args.generation_max_length,
                    )
                    generation = generation[:, input_len:]

                    responses.append(jax.device_get(generation))

            jax.tree.map(lambda x: x.delete(), inference_params)
            del inference_params
            
            
            responses = jnp.concatenate(responses, axis=0)
            
            local_responses = responses
            local_responses = tokenizer.batch_decode(
                local_responses,
                skip_special_tokens=True,
            )
            local_rewards = verify_answers(
                local_responses,
                {k: v[data_start_idx:data_end_idx] for k, v in raw_batch.items()},
                repeat=train_args.group_size,
            )
            local_rewards = np.where(local_rewards, 1, -1)

            multihost_utils.sync_global_devices('post-generation-step')

            responses = multihost_utils.process_allgather(
                responses, tiled=True
            )
            rewards = multihost_utils.process_allgather(
                local_rewards, tiled=True
            )
            
            
            
            penalization = compute_penalization(
                responses,
                tokenizer.eos_token_id,
                max_possible_tokens=responses.shape[-1],
                penalization_margin=inference_args.penalization_margin,
            ) * 0.2            
            penalization = penalization * (rewards > 0)  # only length penalize correct answers
            rewards = rewards - penalization
            
            # if jax.process_index() == 0:
            #     print(f"Penalization: {penalization.shape} | {penalization.tolist()}", flush=True)

            
            opt_state = jax.device_put(
                cpu_opt_state,
                opt_sharding
            )
            del cpu_opt_state
            
            # ------------------------------------------------------------
            # Compute rewards
            # This can be overlapped with generation and done in parallel (before all_gather)
            # but for simplicity and readability, we do separately here
            # ------------------------------------------------------------
            responses = tokenizer.batch_decode(
                responses,
                skip_special_tokens=True,
            )

            # if jax.process_index() == 0:
            #     print(f"Penalization: {penalization.shape} | {penalization.mean()}", flush=True)
            #     print(f"Rewards: {rewards.shape} | {rewards.mean()}", flush=True)
            # multihost_utils.assert_equal(rewards, 'Rewards are not the same across processes')
            rewards = multihost_utils.broadcast_one_to_all(rewards, is_source=jax.process_index() == 0)

            responses = [{'role': 'assistant', 'content': str(resp)} for resp in responses]
            # repeat raw prompts for group size
            raw_prompt = [p for p in raw_prompt for _ in range(train_args.group_size)]
            batch_with_responses = [p +[r] for p, r in zip(raw_prompt, responses)]
            
            # if step % 10 == 0:
            #     if jax.process_index() == 0:
            #         #dump prompt, generation, rewards, penalization
            #         questions = raw_batch['question']
            #         questions = [q for q in questions for _ in range(train_args.group_size)]
            #         ground_truths = raw_batch['answer']
            #         ground_truths = [g for g in ground_truths for _ in range(train_args.group_size)]
            #         with open(f"{train_args.checkpoint_dir}/gen.json", "w") as f:
            #             # for q, g, r, p in zip(questions, responses, rewards, penalization):
            #             #     f.write(json.dumps({'question': q, 'generation': g, 'reward': r.tolist(), 'penalization': p.tolist()}) + "\n")
            #             all_data = [
            #                 {
            #                     'question': q, 
            #                     'generation': g, 
            #                     'reward': r.tolist(), 
            #                     'penalization': p.tolist(),
            #                     'ground_truth': gt,
            #                     'example_with_response': er,
            #                 } for q, g, r, p, gt, er in zip(questions, responses, rewards, penalization, ground_truths, batch_with_responses)]
            #             f.write(json.dumps(all_data, indent=4))
            
            
            train_batch: Dict[str, Any] = train_tokenizer.apply_chat_template(
                batch_with_responses,
                tokenize=True,
                chat_template=qwen2_chat_template,
                enable_thinking=False,
                max_length=train_args.max_length,
                padding='max_length',  # type: ignore
                truncation=True,
                return_tensors='np',
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            
            
            # if step % 10 == 0:
            #     if jax.process_index() == 0:
            #         train_batch_untokenized = train_tokenizer.apply_chat_template(
            #             batch_with_responses,
            #             tokenize=False,
            #             chat_template=qwen2_chat_template,
            #             enable_thinking=False,
            #         )
            #         with open(f"{train_args.checkpoint_dir}/gen_untokenized.json", "w") as f:
            #             f.write(json.dumps(train_batch_untokenized, indent=4))
            
            train_batch = dict(train_batch)
            train_batch['rewards'] = rewards
            grouped_rewards = rewards.reshape((-1, train_args.group_size))
            advantages = (grouped_rewards - grouped_rewards.mean(axis=-1, keepdims=True)) / (grouped_rewards.std(axis=-1, keepdims=True) + 1e-8)
            advantages = advantages.reshape((-1,))
            train_batch['advantages'] = advantages
            
            if jax.process_index() == 0:
                print(f"Avg reward: {rewards.mean()}, Avg advantage: {advantages.mean()}", flush=True)
                print(f"Max reward: {rewards.max()}, Min reward: {rewards.min()}", flush=True)
                print(f"Max advantage: {advantages.max()}, Min advantage: {advantages.min()}", flush=True)            
            
            
            dropout_rngs = jax.random.fold_in(dropout_rng, cur_step)

            multihost_utils.sync_global_devices('pre-train-step')


            with mesh:
                params, opt_state, metrics = p_train_step(params, opt_state, train_batch, dropout_rngs)
            
            is_last_step = (cur_step + 1) == total_train_steps
            checkpoint_manager.save(
                cur_step, items={'model': params, 'optimizer': opt_state}, force=is_last_step
            )
            train_metrics.append(jax.device_get(metrics))
            
            if cur_step % 2 == 0 and cur_step > 0:
                combined_metrics = combine_metrics(train_metrics)
                log_msg = (
                    f"Step... ({cur_step} "
                    f"Loss: {combined_metrics['loss'].mean()}, "
                    f"Avg Reward: {combined_metrics['avg_reward'].mean()}, "
                    f"Avg Removed: {combined_metrics['num_removed'].mean()}, "
                    f"N Updates: {combined_metrics['n_updates'].mean()}, "
                    # f"Skip Batch: {combined_metrics['skip_batch'].mean()}"
                )
                if jax.process_index() == 0:
                    print(log_msg, flush=True)
                train_metrics = []

        epochs.write(
                f"Epoch... ({epoch + 1}/{train_args.num_epochs})"
            )
    
    checkpoint_manager.wait_until_finished()

if __name__ == '__main__':
    main()