"""Process-start shim for the detr_family environment (active only if DEFECTBENCH_TV_SHIM=1).

Why: torchvision>=0.21 dispatches ``Transform.transform``/``make_params``; DEIMv2 (written for
torchvision 0.20.1) defines only ``_transform``/``_get_params`` in a few transforms. D-FINE and
RT-DETRv2 - same code lineage - ship the alias ``transform -> _transform`` upstream. The adapter
adds that alias in the main process, but on Windows DataLoader workers are *spawned* and re-import
the modules, so the alias must be installed at interpreter start. The fallback below only applies
to classes that do NOT define ``transform``/``make_params`` themselves (torchvision's built-ins,
D-FINE and RT-DETRv2 are unaffected).
"""
import os

if os.environ.get("DEFECTBENCH_TV_SHIM") == "1":
    try:
        from torchvision.transforms import v2 as _T

        _orig_transform = _T.Transform.transform
        _orig_make_params = _T.Transform.make_params

        def _transform_fallback(self, inpt, params):
            f = getattr(type(self), "_transform", None)
            if f is not None:
                return f(self, inpt, params)
            return _orig_transform(self, inpt, params)

        def _make_params_fallback(self, flat_inputs):
            f = getattr(type(self), "_get_params", None)
            if f is not None:
                return f(self, flat_inputs)
            return _orig_make_params(self, flat_inputs)

        _T.Transform.transform = _transform_fallback
        _T.Transform.make_params = _make_params_fallback
    except Exception:  # torchvision missing in this interpreter - nothing to do
        pass
