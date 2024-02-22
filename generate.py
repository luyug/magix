import os
import logging

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from tqdm import tqdm, trange
from functools import partial

import jax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as PS
from jax.sharding import NamedSharding
from flax.training.common_utils import onehot

from transformers import AutoTokenizer, AutoConfig
from simple_parsing import ArgumentParser
from simple_parsing.helpers import list_field

import magix
import magix.models

@dataclass
class GenerateArgs:
    prompts: str = None
    batch_size: int = 32
    pad_to_multiple_of: int = 64
    sample: bool = False
    tempearature: float = 0.7
    seed: int = 42
    max_length: int = 512
    model_type: str = 'llama'
    model_name_or_path: str = None
    model_config_name: Optional[str] = None
    tokenizer_name_or_path: str = None
    mesh_shape: List[int] = list_field(1, -1)
    hf_format: bool = False

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
        
    with open(args.prompts, 'r') as f:
        prompts = f.readlines()
    
    rng_key = jax.random.PRNGKey(args.seed)
    
    with open('generated.txt', 'w') as f:
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
                generated = tokenizer.batch_decode(
                    generated, skip_special_tokens=True)
                for g in generated[:batch_size]:
                    f.write(g + '\n')

if __name__ == "__main__":
    main()