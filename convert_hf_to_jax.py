import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import jax
import jax.numpy as jnp

import flax.traverse_util
from transformers import AutoModelForCausalLM, AutoConfig

from tqdm import tqdm

import magix
from magix import (
    save_model_local,
    models
)

import flax

from argparse import ArgumentParser


def rename_key_and_reshape_tensor(
    pt_tuple_key,
    pt_tensor,
    random_flax_state_dict,
    model_prefix: str,
):
    """Rename PT weight names to corresponding Flax weight names and reshape tensor if necessary"""
    def is_key_or_prefix_key_in_dict(key) -> bool:
        """Checks if `key` of `(prefix,) + key` is in random_flax_state_dict"""
        return len(set(random_flax_state_dict) & {key, (model_prefix,) + key}) > 0

    # layer norm
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
    if pt_tuple_key[-1] in ["weight", "gamma"] and is_key_or_prefix_key_in_dict(renamed_pt_tuple_key):
        return renamed_pt_tuple_key, pt_tensor

    # batch norm layer mean
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("mean",)
    if pt_tuple_key[-1] == "running_mean" and not is_key_or_prefix_key_in_dict(pt_tuple_key):
        return renamed_pt_tuple_key, pt_tensor

    # batch norm layer var
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("var",)
    if pt_tuple_key[-1] == "running_var" and not is_key_or_prefix_key_in_dict(pt_tuple_key):
        return renamed_pt_tuple_key, pt_tensor

    # embedding
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("embedding",)
    if pt_tuple_key[-1] == "weight" and is_key_or_prefix_key_in_dict(renamed_pt_tuple_key):
        return renamed_pt_tuple_key, pt_tensor

    # conv layer
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if pt_tuple_key[-1] == "weight" and pt_tensor.ndim == 4 and not is_key_or_prefix_key_in_dict(pt_tuple_key):
        pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
        return renamed_pt_tuple_key, pt_tensor

    # linear layer
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if pt_tuple_key[-1] == "weight" and not is_key_or_prefix_key_in_dict(pt_tuple_key):
        pt_tensor = pt_tensor.T
        return renamed_pt_tuple_key, pt_tensor

    # old PyTorch layer norm weight
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("weight",)
    if pt_tuple_key[-1] == "gamma":
        return renamed_pt_tuple_key, pt_tensor

    # old PyTorch layer norm bias
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("bias",)
    if pt_tuple_key[-1] == "beta":
        return renamed_pt_tuple_key, pt_tensor

    name = None
    if pt_tuple_key[-3::2] == ("parametrizations", "original0"):
        name = pt_tuple_key[-2] + "_g"
    elif pt_tuple_key[-3::2] == ("parametrizations", "original1"):
        name = pt_tuple_key[-2] + "_v"
    if name is not None:
        renamed_pt_tuple_key = pt_tuple_key[:-3] + (name,)
        return renamed_pt_tuple_key, pt_tensor

    return pt_tuple_key, pt_tensor


def combine_layers(layers):
    layer_names = list(layers.keys())
    layer_names.sort(key=lambda x: int(x))
    print('Layer names:', layer_names)
    layers = [layers[name] for name in layer_names]
    layers = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *layers)
    
    return layers

def make_scan_params(params):
    lm_head = params.pop('lm_head')
    embed = params['model'].pop('embed_tokens')
    post_norm = params['model'].pop('norm')
    layers = params['model'].pop('layers')
    layers = combine_layers(layers)
    
    return {
        'embed_tokens': embed,
        'layers': {'blocks': layers},
        'lm_head_with_norm': {'lm_head': lm_head, 'norm': post_norm}
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default='meta-llama/Meta-Llama-3-70B-Instruct')
    parser.add_argument("--model_type", type=str, default='llama')
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--make_scan_param", action='store_true')
    args = parser.parse_args()
    
    
    MODEL_NAME = args.model_name
    MODEL_TYPE = args.model_type
    SAVE_PATH = args.save_path
    MAKE_SCAN_PARAM = args.make_scan_param

    config = AutoConfig.from_pretrained(MODEL_NAME)
    print("Loading model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, config=config)
    print("Model loaded", flush=True)

    mesh = magix.create_device_mesh((1, 1), ('data', 'model'))

    _model_cls = models.CAUSAL_LM_MODEL_MAPPING.get(MODEL_TYPE, None)
    if _model_cls is None:
        raise NotImplementedError(f"Model type {MODEL_TYPE} is not implemented")

    with mesh:
        flax_model = _model_cls(config=config, _do_init=False)
        abs_weight_dict = flax.traverse_util.flatten_dict(flax_model._params_shape_tree)


    sdict = model.state_dict()
    del model
    new_sdict = {}
    old_keys = list(sdict.keys())

    for k in tqdm(old_keys, desc="Converting PyTorch weights to Flax weights"):
        pt_tuple_key = tuple(k.split("."))
        name, tensor = rename_key_and_reshape_tensor(pt_tuple_key, sdict[k], abs_weight_dict, flax_model.base_model_prefix)
        new_sdict[name] = tensor.numpy()
        del sdict[k]

    del sdict
    params = flax.traverse_util.unflatten_dict(new_sdict)


    if MAKE_SCAN_PARAM:
        params = make_scan_params(params)

    save_model_local(params, SAVE_PATH)
