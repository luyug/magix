
import orbax.checkpoint
import numpy as np
import jax
import logging
from functools import partial
from jax.sharding import Mesh

from . import spmd_utils

logger = logging.getLogger(__name__)



def array_restore_args_from_sharding_pytree(pytree):
    return jax.tree_util.tree_map(
        lambda s: orbax.checkpoint.ArrayRestoreArgs(
            restore_type=jax.Array,
            sharding=s,
        ),
        pytree)


def load_model_hub(
    model_cls,
    model_name,
    sharding_config,
    mesh,
    ignore_mismatched_sizes=True,
    half=False,
    from_pt=True,
):  
    # Define sharding function using sharding config over mesh
    get_sharding = partial(
        spmd_utils.get_sharding,
        sharding_config=sharding_config,
        mesh=mesh
    )
    
    # Load model from hub
    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        with Mesh(devices = np.array(jax.local_devices(backend='cpu')[0]).reshape(1,1), axis_names=('data', 'model')):
            model = model_cls.from_pretrained(model_name, ignore_mismatched_sizes=ignore_mismatched_sizes,from_pt=from_pt)
            if not half:
                model.params = model.to_fp32(model.params)
            else:
                model.params = model.to_bf16(model.params)
    logger.info("Model loaded from hub")
    
    
    # Shard model onto device mesh
    model_sharding = jax.tree_util.tree_map_with_path(get_sharding, model.params)
    sharded_params = jax.tree_map(
        lambda a, s: jax.make_array_from_callback(a.shape, s, lambda i: a[i]),
        model.params, model_sharding
    )
    logger.info("Model shards transferred to devices")
    
    return model, sharded_params


def load_model_and_optimizer_local(
    model_cls,
    optimizer,
    checkpoint_manager,
    sharding_config,
    mesh,
    model_name=None,
    model_config=None,
    step=None,
):
    # Create sharding function using sharding config over mesh
    get_sharding = partial(
        spmd_utils.get_sharding,
        sharding_config=sharding_config,
        mesh=mesh
    )
    
    # Load model config from hub
    if model_config is None:
        model_config = model_cls.config_class.from_pretrained(model_name)
    
    # Create model instance and get shape pytrees for model and optimizer
    with Mesh(devices = np.array(jax.devices('cpu')[0]).reshape(1,1), axis_names=('data', 'model')):
        model_no_init = model_cls(model_config, _do_init=False)
    
        def opt_shape():
            params = model_no_init.init_weights(model_no_init.key, model_no_init.input_shape)
            return optimizer.init(params)
        opt_shapes = jax.eval_shape(opt_shape)

    # Define sharding for model and optimizer
    model_sharding = jax.tree_util.tree_map_with_path(get_sharding, model_no_init._params_shape_tree)
    opt_sharding = jax.tree_util.tree_map_with_path(get_sharding, opt_shapes)
    
    # Restore model and optimizer from local storage
    step = checkpoint_manager.latest_step() if step is None else step
    restore_kwargs = {
        'model': {'restore_args': array_restore_args_from_sharding_pytree(model_sharding)},
        'optimizer': {'restore_args': array_restore_args_from_sharding_pytree(opt_sharding)}
    }
    restored = checkpoint_manager.restore(
        checkpoint_manager.latest_step(),
        items={
            'model': model_no_init._params_shape_tree,
            'optimizer': opt_shapes
        }, 
        restore_kwargs=restore_kwargs
    )
    params, opt_state = restored['model'], restored['optimizer']
    logger.info(
        "Model and optimizer restored from local storage at step %d", checkpoint_manager.latest_step())
    
    return model_no_init, params, opt_state


def load_model_local(
    model_cls,
    path,
    sharding_config,
    mesh,
    model_name=None,
    model_config=None,
):
    # Create sharding function using sharding config over mesh
    get_sharding = partial(
        spmd_utils.get_sharding,
        sharding_config=sharding_config,
        mesh=mesh
    )
    
    # Load model config from hub
    if model_config is None:
        model_config = model_cls.config_class.from_pretrained(model_name)
    
    # Create model instance and get shape pytrees for model and optimizer
    with Mesh(devices = np.array(jax.devices('cpu')[0]).reshape(1,1), axis_names=('data', 'model')):
        model_no_init = model_cls(model_config, _do_init=False)
    
    # Define sharding for model and optimizer
    model_sharding = jax.tree_util.tree_map_with_path(get_sharding, model_no_init._params_shape_tree)
    
    # Restore model
    checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    params = checkpointer.restore(
        path,
        restore_args=array_restore_args_from_sharding_pytree(model_sharding)
    )
    logger.info("Model restored from local storage at %s", path)
    
    return model_no_init, params


def save_model_local(
    params,
    path,
):
    checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    checkpointer.save(path, params)
    logger.info("Model saved to local storage at %s", path)


def get_chckpoint_manager(checkpoint_dir, save_steps=500, max_to_keep=3, items=['model', 'optimizer'], json_items=[]):
    options = orbax.checkpoint.CheckpointManagerOptions(
        save_interval_steps=save_steps, max_to_keep=max_to_keep)
    def get_checkpointer():
        return orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    def get_json_checkpointer():
        return orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.JsonCheckpointHandler())
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        checkpoint_dir,
        {item: get_checkpointer() for item in items} | {item: get_json_checkpointer() for item in json_items},
        options
    )
    return checkpoint_manager
