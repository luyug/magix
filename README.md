# Magix
Magix is a mininalist toolkit for training LLM with flexible data and model parallel.

## Features
- Training Billion-scale LLM on GPUs or TPUs.
- Familiar Huggingface interfaces.
- Pre-defined model parallel (sharding) rules for popular models like Llama, Mistral, Gemma, etc.
- Acceleration with flash attention operations and opration fusion.
- Very fast checkpoint save/restore with arbirary device and parallism design.

## Magix 101
If you have ever used Huggingface Flax transformers, using magix is as simple as adding several magic functions with the full workflow preserved.

1. We start by importing necessary dependencies,
```
import magix
from magix.models.llama_model import FlaxLlamaForCausalLM
```

2. The first difference here is that we will explicitly reason about the all the GPU(TPU) devices availabel to us. We will place the GPU in a grid (aka mesh) using the `magix.create_device_mesh` function.
```
# assume we have 4 GPUs in total. we can arrange them arbitrarily
# say, we arrange them into 2x2 mesh and name the first axis `data` and the second axis `model`
# they will be responsible for data and model parallelisms respecticly

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
Here `params` is partitioned and placed on to the mesh. As a side note, JAX will reason about model definition and parameter seperately, analogous to `y = f(x|θ)`

4. For training, you will also need to do something simlar and build the optimizer states onto the mesh,
```
opt_state = magix.initialize_opt_state(optimizer, params, sharding_config, mesh)
```

5. You may have seen tutorial using `jax.pmap`. For our case with both data and model parallelism, we will use instead `jax.jit`,
```
train_step = jax.jit(
    train_step,  # or generate_step
    donate_argnums=...  # set based on the actual function input 
    out_shardings=(magix.item_sharding(params), magix.item_sharding(opt_state),... # set based on the actual function output 
)
```

With all these, you are ready to start your training/inference loop. Take a look at the complete in [train.py](https://github.com/luyug/magix/blob/main/train.py) and [generate.py](https://github.com/luyug/magix/blob/main/generate.py).

## Runnning on GPUs
We recommend using the jax-toolbox jax [container image](https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax) from nvidia.

## Runing on TPUs
Follow the official [install guide](https://jax.readthedocs.io/en/latest/installation.html#pip-installation-google-cloud-tpu) is all you need.
