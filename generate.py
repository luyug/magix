import os
import logging
import json

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from tqdm import tqdm, trange
from functools import partial

import jax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as PS
from jax.sharding import NamedSharding
import orbax.checkpoint

import datasets
from transformers import AutoTokenizer, AutoConfig
from simple_parsing import ArgumentParser
from simple_parsing.helpers import list_field

import magix
import magix.models
import magix.lora

@dataclass
class GenerateArgs:
    prompts: str = None
    use_chat_template: bool = False
    data_field : str = 'prompt'
    hf_data_config: str = None
    hf_data_split: str = 'test'
    output_file: str = 'generated.txt'
    batch_size: int = 32
    pad_to_multiple_of: int = 64
    sample: bool = False
    tempearature: float = 0.7
    seed: int = 42
    max_length: int = 256
    model_type: str = 'llama'
    model_name_or_path: str = None
    model_config_name: Optional[str] = None
    tokenizer_name_or_path: str = None
    mesh_shape: List[int] = list_field(1, -1)
    hf_format: bool = False
    lora: str = None
    lora_alpha: float = 32.0

def main():
    parser = ArgumentParser()
    parser.add_arguments(GenerateArgs, dest="generate_args")
    args = parser.parse_args().generate_args
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        add_eos_token=False,
        use_fast=True,
        padding_side='left',
        legacy=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    _model_cls = magix.models.CAUSAL_LM_MODEL_MAPPING.get(args.model_type)
    if _model_cls is None:
        raise ValueError(f"Model type {args.model_type} not found")

    mesh = magix.create_device_mesh(args.mesh_shape)
        
    if args.hf_format or not os.path.exists(args.model_name_or_path):
        model, params = magix.load_model_hub(
            _model_cls,
            args.model_name_or_path,
            _model_cls.partition_rules,
            mesh,
            half=True,
            from_pt=True,
        )
    else:
        model, params = magix.load_model_local(
            _model_cls,
            args.model_name_or_path,
            _model_cls.partition_rules,
            mesh,
            model_config=AutoConfig.from_pretrained(args.model_config_name),
        )
        
    if args.lora is not None:
        lora = magix.lora.Lora(
            args.lora_alpha,
            rules={
                'layers/.*/kernel': 1,  # rank place holder
            }
        )
        # infer the lora parameters
        lora_params_absract = jax.eval_shape(lora.init_params, jax.random.PRNGKey(0), params)
        lora_params_sharding = magix.lora.create_lora_sharding(_model_cls.partition_rules, mesh, lora_params_absract)
        lora_params = magix.checkpoint_utils.load_by_sharding_no_manager(lora_params_sharding, args.lora)
        params = jax.jit(
            lora.apply,
            donate_argnums=(0,),
            in_shardings=(magix.item_sharding(params), magix.item_sharding(lora_params)),
            out_shardings=magix.item_sharding(params)
            ) (params, lora_params)
        del lora_params
        
    def tokenize(batch):
        return tokenizer(
            batch,
            padding=True, 
            max_length=args.max_length,
            pad_to_multiple_of=args.pad_to_multiple_of,
            truncation=True,
            return_tensors="np",
        )
    
    @partial(
        jax.jit,
        static_argnames=('sample', 'tempearature',),
        out_shardings=NamedSharding(mesh, PS()),
        donate_argnums=(3,)
    )
    def generate(
        params,
        inputs,
        mask,
        rng_key=None,
        sample=False,
        tempearature=1.0,
    ):
        generation = model.generate(
            inputs,
            attention_mask=mask,
            prng_key=rng_key,
            max_length=args.max_length,
            params=params,
            do_sample=sample,
            temperature=tempearature,
        ).sequences
        new_rng_key, _ = jax.random.split(rng_key)
        
        return generation, new_rng_key
    
    if args.prompts.endswith('.txt'):    
        with open(args.prompts, 'r') as f:
            prompts = [l.strip() for l in f]
    elif args.prompts.endswith('.jsonl'):
        with open(args.prompts, 'r') as f:
            prompts = [json.loads(l)[args.data_field] for l in f]
    else:
        prompts = datasets.load_dataset(
            args.prompts, args.hf_data_config
        )[args.hf_data_split][args.data_field]
    
    if args.use_chat_template:
        CHAT_FORMAT = '<|user|>\n{prompt}{eos}<|assistant|>\n'
        prompts = [CHAT_FORMAT.format(prompt=p, eos=tokenizer.eos_token) for p in prompts]
    
    rng_key = jax.random.PRNGKey(args.seed)
    
    with open(args.output_file, 'w') as f:
        with mesh:
            for i in trange(0, len(prompts), args.batch_size):
                batch = prompts[i:i+args.batch_size]
                batch_size = len(batch)
                if batch_size < args.batch_size:
                    batch += ['EMPTY'] * (args.batch_size - len(batch))
                batch = tokenize(batch)
                generated, rng_key = generate(
                    params, 
                    batch['input_ids'],
                    batch['attention_mask'],
                    rng_key,
                    sample=args.sample,
                    tempearature=args.tempearature,
                )
                input_seq_len = batch['input_ids'].shape[1]
                generated = generated[:, input_seq_len:]
                generated = tokenizer.batch_decode(
                    generated, skip_special_tokens=True)
                for p, g in zip(prompts[i:i+batch_size], generated):
                    f.write(json.dumps({'prompt': p, 'generated': g}) + '\n')

if __name__ == "__main__":
    main()
