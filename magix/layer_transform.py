import jax
import flax.linen as nn

def make_scan_fwd_layer(layer_cls, **kwargs):
    class ScanFwdLayer(layer_cls):
        def __call__(self, xx):
            xs = xx[1:]
            return (super(ScanFwdLayer, self).__call__(*xx, **kwargs),) + xs, xx

    return ScanFwdLayer


def make_scan_bwd_layer(layer_cls, **kwargs):
    class ScanBwdLayer(layer_cls):
        def __call__(self, g, saved):
            x, xs = saved[0], saved[1:]
            _, bwd = nn.vjp(
                lambda mdl, x: layer_cls.__call__(mdl, x, *xs, **kwargs), 
                self, 
                x
            )
            params_grad, x_grad = bwd(g)
            return x_grad, params_grad

    return ScanBwdLayer


def make_scan_stack(layer_cls, length=None, remat=False, **layer_kwargs):
    scan_layer = make_scan_fwd_layer(layer_cls, **layer_kwargs)
    if remat:
        scan_layer = nn.remat(
            scan_layer,
            prevent_cse=False,
            # policy=jax.checkpoint_policies.save_only_these_names('q','k', 'v', 'o', 'down_proj'),
            # policy=jax.checkpoint_policies.save_only_these_names('q','k', 'v', 'o', 'down_proj', 'up_proj', 'gate_proj'),
        )
    return nn.scan(
        scan_layer,
        length=length,
        variable_axes={"params": 0, "cache": 0},
        split_rngs={"params": True}
    )