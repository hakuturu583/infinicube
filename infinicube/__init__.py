print("Hello from infinicube")

# Install the fvdb 0.2.0 -> 0.3.0 compatibility shim as early as possible, so it
# patches the fvdb / fvdb.nn namespaces before any voxelgen module imports them.
# Guarded so the CPU-only environments that have no fvdb still import infinicube.
try:
    from .voxelgen import fvdb_compat as _fvdb_compat  # noqa: F401
except Exception as _e:  # pragma: no cover
    import os as _os

    if _os.environ.get("INFINICUBE_DEBUG_SHIM"):
        print(f"[infinicube] fvdb compat shim not installed: {_e!r}")

# diffsynth (video generation) does `from transformers.modeling_utils import
# PretrainedConfig`, but transformers >=5 moved it to configuration_utils.
# Re-export it so diffsynth imports on the pinned-newer transformers.
try:
    import transformers.modeling_utils as _tmu
    import transformers.configuration_utils as _tcu

    if not hasattr(_tmu, "PretrainedConfig"):
        _tmu.PretrainedConfig = _tcu.PretrainedConfig
except Exception:
    pass

from .utils.wds_utils import get_sample
