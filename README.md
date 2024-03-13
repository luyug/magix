# Magix
Magix is a mininalist toolkit for training LLM with flexible data and model parallel.

## Features
- Training Billion-scale LLM on GPUs and TPUs.
- Familiar Huggingface model interfaces and eco-system (dataset, hub, etc.).
- Pre-defined model parallel (sharding) rules for popular models like Llama, Mistral, Gemma, etc.
- Acceleration with flash attention and operation fusion.
- Fast checkpoint save/restore with arbirary device and parallism design.

## Magix 101
If you have ever used Huggingface Flax transformers, using magix is as simple as adding several magic functions into the common worflow.

1. We start by importing necessary dependencies,
```
import magix
from magix.models.llama_model import FlaxLlamaForCausalLM
```

2. We will explicitly reason about all the GPU(TPU) devices available to us. We will place the GPUs in a grid (aka mesh) using the `magix.create_device_mesh` function.
```
# Assume we have 4 GPUs in total; we can arrange them arbitrarily.
# Say, we arrange them into 2x2 mesh and name the first axis `data` and the second axis `model`.
# These axes will be responsible for data and model parallelisms respectively.

mesh = magix.create_device_mesh((2,2), names=('data', 'model'))
```

3. For the next step we will load our model onto the mesh, each device will hold a part (shard) of the full model. Instead of the familiar `from_pretrained`, we will use the function `magix.load_model_hub` function which will call `from_pretrained` internally but also place the model correctly.
```
model, params = magix.load_model_hub(
  FlaxLlamaForCausalLM,
  'meta-llama/Llama-2-13b',
  FlaxLlamaForCausalLM.partition_rules,  # use the pre-defined partitioning
  mesh
)
```
Here `params` is partitioned and placed on to the mesh. As a side note, JAX will reason about model definition and parameter seperately, analogous to `y = f(x|θ)`.

4. For training, you will also need to do something simlar and build the optimizer states onto the mesh,
```
opt_state = magix.initialize_opt_state(optimizer, params, sharding_config, mesh)
```

5. You may have seen tutorial using `jax.pmap`. For our case with both data and model parallelism, we will use the more powerful `jax.jit`,
```
train_step = jax.jit(
    train_step,  # or generate_step
    donate_argnums=...  # set based on the actual function input 
    out_shardings=(magix.item_sharding(params), magix.item_sharding(opt_state),... # set based on the actual function output 
)
```

With all these, you are ready to start your training/inference loop.

Take a look at the complete scripts in [train.py](https://github.com/luyug/magix/blob/main/train.py), [train_lora.py](https://github.com/luyug/magix/blob/main/train_lora.py) and [generate.py](https://github.com/luyug/magix/blob/main/generate.py).

## Example: Train a Mistral ChatBot with Lora and Data&Tensor Parallelism
Assume we have 4 GPUs. Let's train `mistral-7b` on `UltraChat` with data and tensor parallism, `dp=2` and `tp=2`:
```
python train_lora.py \
    --checkpoint_dir /absolute/path/to/checkpoint \
    --model_type mistral \
    --model_name mistralai/Mistral-7B-v0.1 \
    --tokenizer_name mistralai/Mistral-7B-v0.1 \
    --train_file HuggingFaceH4/ultrachat_200k \
    --split train_sft \
    --train_data_field messages \
    --use_chat_template \
    --batch_size 32 \
    --num_epochs 1 \
    --learning_rate 5e-5 \
    --seed 12345 \
    --mesh_shape 2 2  \
    --weight_decay 0.001 \
    --max_length 1024
```
After training, let's solve some math problems. Do generation with full tensor parallel `tp=4`:
```
python generate.py \
    --prompts gsm8k \
    --hf_data_config main \
    --hf_data_split test \
    --use_chat_template \
    --data_field question \
    --output_file generation.jsonl \
    --mesh_shape 1 -1  \
    --model_type llama \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --tokenizer_name_or_path mistralai/Mistral-7B-v0.1 \
    --model_config_name mistralai/Mistral-7B-v0.1 \
    --batch_size 32 \
    --pad_to_multiple_of 64 \
    --max_length 512 \
    --lora /absolute/path/to/checkpoint/EVALUATION_STEP/lora
```

## Runnning on GPUs
We recommend using the jax-toolbox jax [container image](https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax) from nvidia. We have example [Dockerfile](https://github.com/luyug/magix/blob/main/container/Dockerfile) and Singulrity [Definition File](https://github.com/luyug/magix/blob/main/container/magix-gpu.def).

## Runing on TPUs
Follow the official [install guide](https://jax.readthedocs.io/en/latest/installation.html#pip-installation-google-cloud-tpu) is all you need.
