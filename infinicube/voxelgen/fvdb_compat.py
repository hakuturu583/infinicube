# SPDX-License-Identifier: Apache-2.0
"""fvdb 0.2.0 -> fvdb-core 0.3.0 compatibility shim.

InfiniCube's voxelgen was written against fvdb 0.2.0 (the ``VDBTensor``-centric
API with free ``gridbatch_from_*`` functions and ``fvdb.nn`` modules that consume
and return ``VDBTensor``). fvdb-core 0.3.0 (the only version that builds on
Blackwell / sm_120) reorganised that API: ``VDBTensor`` was removed, factory
functions became ``GridBatch`` classmethods, and ``fvdb.nn`` modules operate on
``(JaggedTensor, GridBatch)`` and (for conv) a ``ConvolutionPlan``.

Importing this module monkeypatches the 0.2.0 surface back onto the live ``fvdb``
and ``fvdb.nn`` namespaces, so InfiniCube runs unmodified. Import it BEFORE any
voxelgen module (done from ``infinicube/__init__.py``).

Mapping decisions are recorded in ``SHIM_PORT_NOTES.md``.
"""
from __future__ import annotations

import torch
import torch.nn as tnn

import fvdb
from fvdb import ConvolutionPlan, GridBatch, JaggedTensor
from fvdb import nn as fvnn


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_jagged(x):
    """Wrap a raw torch tensor as a single-grid JaggedTensor; pass JaggedTensor through."""
    if isinstance(x, JaggedTensor):
        return x
    if isinstance(x, torch.Tensor):
        return JaggedTensor([x])
    return _wrap_jt(x)


def _coerce_vec(v):
    """fvdb-core's strict type checks reject Python lists (esp. lists of 0-dim
    tensors, which InfiniCube builds for voxel_sizes/origins). Convert to a tensor."""
    if isinstance(v, (list, tuple)):
        try:
            if len(v) > 0 and isinstance(v[0], torch.Tensor):
                return torch.stack([torch.as_tensor(x).flatten()[0] for x in v]).float()
            return torch.tensor(v, dtype=torch.float32)
        except Exception:
            return v
    return v


def _wrap_jt(x):
    """Ensure ``x`` is a *Python* JaggedTensor. The native ``fvdb.jcat`` returns a
    raw C++ JaggedTensor impl for the JaggedTensor branch, which the Python API
    then rejects on the next call; re-wrap those here."""
    if isinstance(x, JaggedTensor):
        return x
    try:
        return JaggedTensor(impl=x)
    except Exception:
        return x


# --------------------------------------------------------------------------- #
# VDBTensor: light container bundling (grid, features, conv-plan cache)
# --------------------------------------------------------------------------- #
class VDBTensor:
    __slots__ = ("grid", "data", "kmap")

    def __init__(self, grid: GridBatch, data, kmap=None):
        if isinstance(data, torch.Tensor):
            data = grid.jagged_like(data)
        elif not isinstance(data, JaggedTensor):
            data = _wrap_jt(data)
        self.grid = grid
        self.data = data  # JaggedTensor
        self.kmap = kmap

    # --- attribute compatibility ------------------------------------------ #
    @property
    def jdata(self) -> torch.Tensor:
        return self.data.jdata

    @property
    def feature(self) -> JaggedTensor:
        return self.data

    @property
    def jidx(self) -> torch.Tensor:
        return self.data.jidx

    @property
    def device(self):
        return self.data.jdata.device

    @property
    def dtype(self):
        return self.data.jdata.dtype

    def to(self, *args, **kwargs):
        return VDBTensor(self.grid, self.grid.jagged_like(self.jdata.to(*args, **kwargs)), self.kmap)

    def type(self, dtype):
        return VDBTensor(self.grid, self.grid.jagged_like(self.jdata.type(dtype)), self.kmap)

    def _wrap(self, new_jdata: torch.Tensor) -> "VDBTensor":
        return VDBTensor(self.grid, self.grid.jagged_like(new_jdata), self.kmap)

    def to_dense(self) -> torch.Tensor:
        """0.2.0 VDBTensor.to_dense(): sparse features -> dense (B, X, Y, Z, C).

        Uses the grid's minimum ijk as the dense origin so it round-trips with
        ``grid.read_from_dense(..., dense_origins=min(ijk))`` used by the dense UNet.
        """
        min_coord = torch.min(self.grid.ijk.jdata, dim=0)[0]
        return self.grid.inject_to_dense_cminor(self.data, min_coord=min_coord)

    # --- elementwise arithmetic (0.2.0 ElementwiseMixin) ------------------ #
    @staticmethod
    def _other(o):
        return o.jdata if isinstance(o, VDBTensor) else o

    def __add__(self, o):
        return self._wrap(self.jdata + self._other(o))

    def __radd__(self, o):
        return self._wrap(self._other(o) + self.jdata)

    def __sub__(self, o):
        return self._wrap(self.jdata - self._other(o))

    def __rsub__(self, o):
        return self._wrap(self._other(o) - self.jdata)

    def __mul__(self, o):
        return self._wrap(self.jdata * self._other(o))

    def __rmul__(self, o):
        return self._wrap(self._other(o) * self.jdata)

    def __truediv__(self, o):
        return self._wrap(self.jdata / self._other(o))

    def __neg__(self):
        return self._wrap(-self.jdata)

    def __repr__(self):
        return f"VDBTensor(grid_count={self.grid.grid_count}, jdata={tuple(self.jdata.shape)})"


# --------------------------------------------------------------------------- #
# nn modules: wrap 0.3.0 (data, grid[, plan]) modules to speak VDBTensor
# --------------------------------------------------------------------------- #
class SparseConv3d(fvnn.SparseConv3d):
    """0.2.0-style: ``y = conv(x)`` where x, y are VDBTensor.

    Builds a ConvolutionPlan on the fly (stride==1 -> same grid, stride>1 ->
    coarsened target grid) and delegates to the 0.3.0 conv (which adds bias).
    """

    def forward(self, x: VDBTensor, out_grid: GridBatch | None = None) -> VDBTensor:
        src_grid = x.grid
        stride = self.stride
        is_unit_stride = bool(torch.all(stride == 1))
        # 1x1x1 conv is just a per-voxel linear; fvdb refuses to build a plan for it.
        if self.kernel_volume == 1 and is_unit_stride:
            out = x.jdata @ self.weight.t()
            if self.bias is not None:
                out = out + self.bias
            return VDBTensor(src_grid, src_grid.jagged_like(out), kmap=x.kmap)
        if out_grid is not None:
            target_grid = out_grid
        elif is_unit_stride:
            target_grid = src_grid
        else:
            target_grid = src_grid.coarsened_grid(stride)
        plan = ConvolutionPlan.from_grid_batch(
            self.kernel_size, self.stride, src_grid, target_grid
        )
        out_data = super().forward(x.data, plan)
        return VDBTensor(target_grid, out_data, kmap=x.kmap)


class SparseConvTranspose3d(fvnn.SparseConvTranspose3d):
    def forward(self, x: VDBTensor, out_grid: GridBatch | None = None) -> VDBTensor:
        src_grid = x.grid
        target_grid = out_grid if out_grid is not None else src_grid
        plan = ConvolutionPlan.from_grid_batch_transposed(
            self.kernel_size, self.stride, src_grid, target_grid
        )
        out_data = super().forward(x.data, plan)
        return VDBTensor(target_grid, out_data, kmap=x.kmap)


class GroupNorm(fvnn.GroupNorm):
    def forward(self, x: VDBTensor) -> VDBTensor:
        out = super().forward(x.data, x.grid)
        return VDBTensor(x.grid, out, x.kmap)


class BatchNorm(fvnn.BatchNorm):
    def forward(self, x: VDBTensor) -> VDBTensor:
        out = super().forward(x.data, x.grid)
        return VDBTensor(x.grid, out, x.kmap)


def _ref_grid(ref):
    """A grid reference may be a VDBTensor (use .grid) or a bare GridBatch."""
    if ref is None:
        return None
    if isinstance(ref, VDBTensor):
        return ref.grid
    return ref  # already a GridBatch


class AvgPool(fvnn.AvgPool):
    def forward(self, x: VDBTensor, ref_coarse_data=None) -> VDBTensor:
        out_data, out_grid = super().forward(x.data, x.grid, _ref_grid(ref_coarse_data))
        return VDBTensor(out_grid, out_data)


class MaxPool(fvnn.MaxPool):
    def forward(self, x: VDBTensor, ref_coarse_data=None) -> VDBTensor:
        out_data, out_grid = super().forward(x.data, x.grid, _ref_grid(ref_coarse_data))
        return VDBTensor(out_grid, out_data)


class UpsamplingNearest(fvnn.UpsamplingNearest):
    def forward(self, x: VDBTensor, ref=None) -> VDBTensor:
        """0.2.0 accepted either a target VDBTensor (``ref_fine_data``, use its grid)
        or a coarse-grid structure mask (JaggedTensor). fvdb-core's ``refine`` mask
        is coarse-grid, so a mask passes straight through."""
        mask = None
        fine_grid = None
        if isinstance(ref, VDBTensor):
            fine_grid = ref.grid
        elif isinstance(ref, GridBatch):
            fine_grid = ref
        elif ref is not None:
            mask = ref if isinstance(ref, JaggedTensor) else _wrap_jt(ref)
        out_data, out_grid = super().forward(x.data, x.grid, mask, fine_grid)
        return VDBTensor(out_grid, out_data)


# ---- torch-backed per-voxel modules (operate on .jdata) ------------------- #
class Linear(tnn.Linear):
    def forward(self, x):
        if isinstance(x, VDBTensor):
            return x._wrap(super().forward(x.jdata))
        return super().forward(x)


class _Activation:
    """Mixin: apply a torch activation to VDBTensor.jdata or a raw tensor."""

    def forward(self, x):
        if isinstance(x, VDBTensor):
            return x._wrap(super().forward(x.jdata))
        return super().forward(x)


class SiLU(_Activation, tnn.SiLU):
    pass


class ReLU(_Activation, tnn.ReLU):
    pass


class LeakyReLU(_Activation, tnn.LeakyReLU):
    pass


class GELU(_Activation, tnn.GELU):
    pass


class Dropout(_Activation, tnn.Dropout):
    pass


class Identity(tnn.Identity):
    def forward(self, x):
        return x


def vdbtensor_from_dense(dense: torch.Tensor, ijk_min=(0, 0, 0), voxel_sizes=1, origins=0):
    """0.2.0 fvdb.nn.vdbtensor_from_dense: build a dense grid + features from a
    dense ``(B, X, Y, Z, C)`` tensor. Returns a VDBTensor whose grid covers the
    full dense box and whose features are the flattened active-voxel values.
    """
    B, X, Y, Z, C = dense.shape
    grid = GridBatch.from_dense(
        num_grids=B,
        dense_dims=[X, Y, Z],
        ijk_min=list(ijk_min),
        voxel_sizes=voxel_sizes,
        origins=origins,
        device=dense.device,
    )
    feat = grid.inject_from_dense_cminor(dense, dense_origins=list(ijk_min))
    return VDBTensor(grid, feat)


class FillFromGrid(tnn.Module):
    """0.2.0 fvnn.FillFromGrid: scatter src VDBTensor features onto a target grid."""

    def __init__(self, default_value: float = 0.0):
        super().__init__()
        self.default_value = default_value

    def forward(self, src: VDBTensor, target) -> VDBTensor:
        target_grid = _ref_grid(target)
        out = target_grid.inject_from(src.grid, src.data, default_value=self.default_value)
        return VDBTensor(target_grid, out)


# --------------------------------------------------------------------------- #
# free functions
# --------------------------------------------------------------------------- #
def gridbatch_from_points(points, voxel_sizes=1, origins=0, device=None):
    return GridBatch.from_points(
        _as_jagged(points), _coerce_vec(voxel_sizes), _coerce_vec(origins), device=device
    )


def gridbatch_from_ijk(ijk, voxel_sizes=1, origins=0, device=None):
    return GridBatch.from_ijk(
        _as_jagged(ijk), _coerce_vec(voxel_sizes), _coerce_vec(origins), device=device
    )


def gridbatch_from_nearest_voxels_to_points(points, voxel_sizes=1, origins=0, device=None):
    return GridBatch.from_nearest_voxels_to_points(
        _as_jagged(points), _coerce_vec(voxel_sizes), _coerce_vec(origins), device=device
    )


def gridbatch_from_mesh(vertices, faces, voxel_sizes=1, origins=0, device=None):
    return GridBatch.from_mesh(
        _as_jagged(vertices), _as_jagged(faces),
        _coerce_vec(voxel_sizes), _coerce_vec(origins), device=device
    )


def sparse_grid_from_mesh(vertices, faces, voxel_sizes=1, origins=0, device=None):
    return GridBatch.from_mesh(
        _as_jagged(vertices), _as_jagged(faces),
        _coerce_vec(voxel_sizes), _coerce_vec(origins), device=device
    )


def gridbatch_from_dense(
    num_grids=1, dense_dims=None, ijk_min=0, voxel_sizes=1, origins=0, mask=None, device=None
):
    return GridBatch.from_dense(
        num_grids, dense_dims, ijk_min, _coerce_vec(voxel_sizes), _coerce_vec(origins),
        mask=mask, device=device
    )


_ORIG_JCAT = fvdb.jcat  # capture the real 0.3.0 jcat before we patch the namespace


def jcat(things_to_cat, dim=None):
    """0.2.0-compatible jcat that also understands VDBTensor lists.

    - list of VDBTensor, dim is None -> batch-concatenate (combine grids + data)
    - list of VDBTensor, dim given   -> feature-concatenate on the shared grid
    - list of GridBatch / JaggedTensor -> delegate to the native jcat
    """
    seq = list(things_to_cat)
    if len(seq) > 0 and any(isinstance(t, VDBTensor) for t in seq):
        datas = [t.data if isinstance(t, VDBTensor) else _wrap_jt(t) for t in seq]
        grid0 = next(t.grid for t in seq if isinstance(t, VDBTensor))
        if dim is None:
            grid = _ORIG_JCAT([t.grid for t in seq if isinstance(t, VDBTensor)])
            return VDBTensor(grid, _wrap_jt(_ORIG_JCAT(datas)))
        return VDBTensor(grid0, _wrap_jt(_ORIG_JCAT(datas, dim)))
    result = _ORIG_JCAT(seq, dim)
    return _wrap_jt(result) if not isinstance(seq[0], GridBatch) else result


def cat(things_to_cat, dim=None):
    return jcat(things_to_cat, dim)


# --------------------------------------------------------------------------- #
# GridBatch method aliases / back-compat (patched onto the class)
# --------------------------------------------------------------------------- #
def _install_gridbatch_compat():
    GB = GridBatch

    # 0.2.0 allowed `GridBatch("cuda")` / `GridBatch(device=...)` to make an empty
    # grid; 0.3.0's __init__ only takes `impl=`. Wrap it to support both.
    if not getattr(GB.__init__, "_infinicube_ctor", False):
        _orig_init = GB.__init__

        def _init(self, device=None, *, impl=None):
            if impl is not None:
                _orig_init(self, impl=impl)
                return
            dev = device if device is not None else "cpu"
            if isinstance(dev, str) and dev == "cuda":
                dev = torch.device("cuda", torch.cuda.current_device())
            empty = GB.from_ijk(
                JaggedTensor([torch.zeros((0, 3), dtype=torch.int32, device=dev)]),
                device=dev,
            )
            _orig_init(self, impl=empty._impl)

        _init._infinicube_ctor = True
        GB.__init__ = _init

    # renamed coordinate transforms
    if not hasattr(GB, "grid_to_world"):
        GB.grid_to_world = lambda self, ijk: self.voxel_to_world(_as_jagged(ijk))
    if not hasattr(GB, "world_to_grid"):
        GB.world_to_grid = lambda self, xyz: self.world_to_voxel(_as_jagged(xyz))

    # fill_from_grid(src_data, src_grid, default) -> inject_from(src_grid, src, default)
    if not hasattr(GB, "fill_from_grid"):
        def fill_from_grid(self, src_data, src_grid, default_value=0.0):
            return self.inject_from(src_grid, _as_jagged(src_data), default_value=default_value)
        GB.fill_from_grid = fill_from_grid

    # subdivided_grid(factor): coarse->fine grid structure only
    if not hasattr(GB, "subdivided_grid"):
        def subdivided_grid(self, subdiv_factor):
            dummy = self.jagged_like(torch.zeros(self.total_voxels, 1, device=self.device))
            _, fine = self.refine(subdiv_factor, dummy)
            return fine
        GB.subdivided_grid = subdivided_grid

    # set_from_ijk: 0.2.0 mutated an empty grid in place; rebuild _impl here.
    if not hasattr(GB, "set_from_ijk"):
        def set_from_ijk(self, ijk, voxel_sizes=1, origins=0):
            new = GridBatch.from_ijk(
                _as_jagged(ijk), _coerce_vec(voxel_sizes), _coerce_vec(origins),
                device=self.device,
            )
            self._impl = new._impl
            return self
        GB.set_from_ijk = set_from_ijk

    # read_from_dense alias
    if not hasattr(GB, "read_from_dense"):
        GB.read_from_dense = lambda self, dense, dense_origins=0: self.inject_from_dense_cminor(
            dense, dense_origins=dense_origins
        )

    # fvdb-core requires a device with an explicit index; a bare "cuda" /
    # torch.device("cuda") raises "Device must specify an index". Normalise in .to().
    _orig_to = getattr(GB, "to", None)
    if _orig_to is not None and not getattr(_orig_to, "_infinicube_to", False):
        def _to(self, device):
            if isinstance(device, str):
                device = torch.device(device)
            if isinstance(device, torch.device) and device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            return _orig_to(self, device)
        _to._infinicube_to = True
        GB.to = _to

    # fvdb-core's GridBatch.cuda() calls impl.cuda() which requires an explicit
    # device index; 0.2.0 accepted a bare .cuda(). Route through .to(cuda:idx).
    _orig_cuda = getattr(GB, "cuda", None)
    if _orig_cuda is not None:
        def _cuda(self, device=None):
            if device is None:
                device = torch.device("cuda", torch.cuda.current_device())
            elif isinstance(device, (int,)):
                device = torch.device("cuda", device)
            return self.to(device)
        GB.cuda = _cuda

    # renamed containment query
    if not hasattr(GB, "points_in_active_voxel") and hasattr(GB, "points_in_grid"):
        GB.points_in_active_voxel = lambda self, points: self.points_in_grid(_as_jagged(points))

    # 0.2.0 accepted raw torch tensors where 0.3.0 requires a JaggedTensor.
    # Wrap the coordinate/query methods so raw (N,3)/(N,) tensors still work.
    for _name in ("ijk_to_index", "voxel_to_world", "world_to_voxel",
                  "points_in_active_voxel", "points_in_grid", "coords_in_grid",
                  "splat_trilinear", "sample_trilinear", "ijk_to_inv_index",
                  "segments_along_rays", "voxels_along_rays", "uniform_ray_samples"):
        _orig = getattr(GB, _name, None)
        if _orig is None:
            continue

        def _make(orig):
            def _wrapped(self, *args, **kwargs):
                args = tuple(_as_jagged(a) if isinstance(a, torch.Tensor) else a for a in args)
                return orig(self, *args, **kwargs)
            return _wrapped

        setattr(GB, _name, _make(_orig))


def _patched_gridbatch_ctor():
    """Allow ``fvdb.GridBatch(device=...)`` (empty grid) as in 0.2.0."""
    orig_new = GridBatch.__new__

    def __new__(cls, *args, device=None, impl=None, **kwargs):
        if impl is not None:
            return orig_new(cls)
        # build an empty single grid
        empty = GridBatch.from_ijk(
            JaggedTensor([torch.zeros((0, 3), dtype=torch.int32, device=device or "cpu")]),
            device=device,
        )
        return empty

    # only wrap if the ctor doesn't already accept device
    # (kept minimal: InfiniCube only uses GridBatch(device=...))
    return __new__


# --------------------------------------------------------------------------- #
# install everything
# --------------------------------------------------------------------------- #
def _patch_torch_load():
    """torch>=2.6 defaults ``weights_only=True``; InfiniCube's Lightning checkpoints
    contain omegaconf/config objects. Default back to full unpickling (checkpoints
    here are trusted, locally-downloaded files)."""
    if getattr(torch.load, "_infinicube_patched", False):
        return
    _orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    _load._infinicube_patched = True
    torch.load = _load


def install():
    _patch_torch_load()
    # nn modules
    fvnn.VDBTensor = VDBTensor
    for name, cls in {
        "SparseConv3d": SparseConv3d,
        "SparseConvTranspose3d": SparseConvTranspose3d,
        "GroupNorm": GroupNorm,
        "BatchNorm": BatchNorm,
        "AvgPool": AvgPool,
        "MaxPool": MaxPool,
        "UpsamplingNearest": UpsamplingNearest,
        "Linear": Linear,
        "SiLU": SiLU,
        "ReLU": ReLU,
        "LeakyReLU": LeakyReLU,
        "GELU": GELU,
        "Dropout": Dropout,
        "Identity": Identity,
        "FillFromGrid": FillFromGrid,
    }.items():
        setattr(fvnn, name, cls)
    fvnn.vdbtensor_from_dense = vdbtensor_from_dense

    # top-level free functions + VDBTensor
    fvdb.VDBTensor = VDBTensor
    fvdb.gridbatch_from_points = gridbatch_from_points
    fvdb.gridbatch_from_ijk = gridbatch_from_ijk
    fvdb.gridbatch_from_nearest_voxels_to_points = gridbatch_from_nearest_voxels_to_points
    fvdb.gridbatch_from_mesh = gridbatch_from_mesh
    fvdb.gridbatch_from_dense = gridbatch_from_dense
    fvdb.sparse_grid_from_mesh = sparse_grid_from_mesh
    fvdb.cat = cat
    fvdb.jcat = jcat

    _install_gridbatch_compat()


install()
