import re
import logging

from functools import partial
from contextlib import nullcontext

import numpy as np
import jax
import jax.experimental.multihost_utils as mhu
from jax.sharding import PartitionSpec as PS
from jax.sharding import NamedSharding, Mesh
from jax.experimental import mesh_utils
from jax._src.tree_util import GetAttrKey

logger = logging.getLogger(__name__)


def get_sharding(k, v, sharding_config=None, mesh=None):
    def get_key(x):
        if isinstance(x, GetAttrKey):
            name = str(x)[1:]
        else:
            name = str(getattr(x, 'key', getattr(x, 'idx', None)))
        return name
    
    path = '/'.join([get_key(p) for p in k])
    rule = PS(None)
    for param_name, sharding_rule in sharding_config.items():
        if re.search(param_name, path):
            rule = sharding_rule
            break
    if len(v.shape) == 0:
        rule = PS()
    
    if mesh is None:
        return rule

    return NamedSharding(mesh, rule)


def get_sharding_tree(pytree, sharding_config, mesh=None):
    return jax.tree_util.tree_map_with_path(
        partial(get_sharding, sharding_config=sharding_config, mesh=mesh),
        pytree
    )


def item_sharding(pytree):
    return jax.tree_map(lambda x: x.sharding, pytree)


def initialize_opt_state(optimizer, sharded_params, sharding_config, mesh):
    get_sharding_fn = partial(
        get_sharding,
        sharding_config=sharding_config,
        mesh=mesh
    )
    opt_shapes = jax.eval_shape(optimizer.init, sharded_params)
    opt_sharding = jax.tree_util.tree_map_with_path(get_sharding_fn, opt_shapes)
    opt_state =  jax.jit(optimizer.init, out_shardings=opt_sharding)(sharded_params)
    logger.info("Optimizer shards initialized on devices")
    return opt_state


def create_device_mesh(shape, names=('data', 'model')):
    if -1 in shape:
        from collections import Counter
        assert Counter(shape)[-1] == 1, "Only one -1 is allowed in shape"
        shape = np.array(jax.devices()).reshape(shape).shape

    return Mesh(devices=mesh_utils.create_device_mesh(shape), axis_names=names)


def duplicate_over(ob, *dup_axes):
    def transform_spec(spec):
        new_axes = [ax if ax not in dup_axes else None for ax in spec]
        return PS(*new_axes)
    return jax.tree_map(transform_spec, ob, is_leaf=lambda x: isinstance(x, PS))


def sub_mesh_map(
    f, 
    global_mesh,
    local_mesh,
    global_specs,
    local_specs,
    out_specs,
):
    def sub_mesh_fn(*args, **kwargs):
        host_args = mhu.global_array_to_host_local_array(
            args,
            global_mesh, 
            global_specs
        )
            
        local_mesh_args = mhu.host_local_array_to_global_array(
            host_args,
            local_mesh,
            local_specs
        )
        
        with local_mesh:
            yy = f(*local_mesh_args, **kwargs)
        
        yy = mhu.global_array_to_host_local_array(yy, local_mesh, out_specs)
        yy = mhu.host_local_array_to_global_array(yy, global_mesh, out_specs)
        
        return yy
    
    return sub_mesh_fn


def create_local_mesh(n_local_mesh, local_mesh_shape, names=('data', 'model')):
    n_processes_local_mesh = jax.process_count() // n_local_mesh
    local_mesh_size = len(jax.devices()) // n_local_mesh
    assert local_mesh_size == np.prod(local_mesh_shape)
    sub_mesh_id = jax.process_index() // n_processes_local_mesh
    
    local_mesh_devices = jax.devices()[sub_mesh_id * local_mesh_size: (sub_mesh_id + 1) * local_mesh_size]
    
    return Mesh(
        devices=mesh_utils.create_device_mesh(local_mesh_shape, local_mesh_devices),
        axis_names=names
        )