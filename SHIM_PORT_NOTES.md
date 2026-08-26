# fvdb 0.2.0 → fvdb-core 0.3.0 compatibility port

InfiniCube's `voxelgen` was written against **fvdb 0.2.0** (VDBTensor-centric API).
The only fvdb that builds on the **Blackwell / sm_120** GPU is **fvdb-core 0.3.0**,
whose API changed substantially. A compatibility shim
(`infinicube/voxelgen/fvdb_compat.py`, installed early from `infinicube/__init__.py`)
recreates the 0.2.0 surface on top of 0.3.0 so InfiniCube runs (almost) unmodified.

## Environment prerequisites (critical)
Runs need, in addition to the venv:
```
export CUDA_HOME=/home/kataoka/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=<venv>/lib/python3.10/site-packages/torch/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```
(see `scratchpad_env.sh`). **The torch/lib entry is mandatory** — see the libc10 note.

### The `libc10.so.1.8` landmine (was the segfault cause)
A stray system `libtorch-dev 1.8.1` lives in `/usr/lib/x86_64-linux-gnu`. fvdb's
`libfvdb.so` ends up with a `NEEDED libc10.so.1.8` entry that, without the venv torch
lib dir on the path, resolves to **system torch 1.8**, mixing two torch runtimes →
segfault in `GridBatchImpl`/`InlineDeviceGuard`. Fix applied:
`ln -s libc10.so <venvtorch>/lib/libc10.so.1.8` (redirects the mislabeled soname to
torch 2.7's own libc10) **and** put `<venvtorch>/lib` first on `LD_LIBRARY_PATH`.

### pybind11
fvdb-core must be built with **pybind11 2.13.6** (what torch 2.7 bundles); building
with pybind11 3.x causes an ABI-incompatible exception-translation crash.

## API mappings implemented in the shim
| 0.2.0 | 0.3.0 handling |
|---|---|
| `fvdb.nn.VDBTensor(grid, data, kmap)` | shim container class; `.jdata/.data/.grid/.kmap/.feature`, elementwise `+ - * /`, `.to_dense()` |
| `fvdb.gridbatch_from_points/ijk/dense/mesh/nearest_voxels_to_points` | `GridBatch.from_*` (raw tensors auto-wrapped to `JaggedTensor`) |
| `fvdb.sparse_grid_from_mesh` | `GridBatch.from_mesh` |
| `fvdb.cat` / `fvdb.jcat` | native `jcat`, VDBTensor-aware (batch-concat grids+data when `dim is None`, feature-concat on shared grid otherwise). **Native jcat returns a raw C++ JaggedTensor impl → always re-wrapped.** |
| `grid.grid_to_world` / `world_to_grid` | `voxel_to_world` / `world_to_voxel` |
| `grid.points_in_active_voxel` | `points_in_grid` |
| `grid.fill_from_grid(src_data, src_grid, default)` | `inject_from(src_grid, src, default_value=)` |
| `grid.set_from_ijk` | rebuild `_impl` via `from_ijk` |
| `grid.subdivided_grid(f)` | `refine(f, dummy)` → grid |
| `grid.read_from_dense` | `inject_from_dense_cminor` |
| `VDBTensor.to_dense()` | `grid.inject_to_dense_cminor(data, min_coord=min(ijk))` — **cminor = channels-last (B,X,Y,Z,C)**, matches 0.2.0 |
| `fvdb.nn.{GroupNorm,BatchNorm,AvgPool,MaxPool}` | wrap 0.3.0 `(data,grid)`→`JaggedTensor` modules to speak VDBTensor |
| `fvdb.nn.SparseConv3d` | build `ConvolutionPlan.from_grid_batch` per forward; **1×1×1 conv = per-voxel linear** (fvdb refuses a plan for it) |
| `fvdb.nn.UpsamplingNearest(x, ref)` | `ref`=VDBTensor→fine_grid; `ref`=JaggedTensor→coarse structure **mask** (fvdb-core `refine`'s mask is coarse-grid, verified empirically) |
| `fvdb.nn.{Linear,SiLU,ReLU,LeakyReLU,Dropout,GELU,Identity}` | torch modules that also accept VDBTensor (apply on `.jdata`) |
| `fvdb.nn.vdbtensor_from_dense` | `from_dense` + `inject_from_dense_cminor` |

## InfiniCube edits (surgical, non-fvdb)
- `infinicube/__init__.py`: import the shim early (guarded so CPU-only envs still work).
- `infinicube/voxelgen/utils/extrap_util.py::transform_points`: compute in the matrix
  dtype and cast back (torch ≥2 forbids mixed-dtype matmul).
- Also relied on the shim patching `torch.load` to `weights_only=False` (torch ≥2.6
  default broke Lightning checkpoint loading of omegaconf configs).

## Numerical fidelity
**Trustworthy for Step 1.** The pretrained `voxel_diffusion.ckpt` conv weights are
`(Do,Di,3,3,3)` = fvdb-core's own `SparseConv3d.weight` shape, load directly, and the
generated Kashiwanoha voxel world is **semantically correct** — gray road with yellow
lane lines matching the map condition, trees, terrain (see
`visualization/.../kashiwanoha/0.jpg`). Conv semantics are shared fvdb lineage;
`to_dense`/`read_from_dense` round-trip verified via correct output. Steps 2/3 (video,
gaussian) exercise more of the shim (SparseConvTranspose, voxel_branch) and are not yet
validated.

## Steps 2 & 3 (guidance buffer + Wan video + 3D Gaussians + render) — additional fixes

**Reached the final goal: 3D Gaussians generated and RENDERED.**
- Render: `visualization/.../gaussian_scene_generation/.../visualize_gsm/static_pd_images.jpg`
  (8 novel views + front view); clean front-view crop: `visualization/kashiwanoha_3dgs_render.jpg`.
- Verified genuine: predicted render differs from the GT input frame (mean|Δ|≈15/255, not identical).

### Extra shim gaps patched (fvdb_compat.py)
- `GridBatch("cuda")` / `GridBatch(device=...)` empty-grid ctor (0.3.0 only takes `impl=`).
- `GridBatch.cuda()` / `.to("cuda")` require an explicit device index → normalise to `cuda:current`.
- Auto-wrap raw tensors for `segments_along_rays`, `voxels_along_rays`, `uniform_ray_samples`, `points_in_grid`.
- `_coerce_vec`: convert list/list-of-0dim-tensor `voxel_sizes`/`origins` to a tensor (0.3.0's strict `NumericMaxRank2` rejects Python lists) — applied in all `gridbatch_from_*`.
- Pools / `UpsamplingNearest` / `FillFromGrid`: accept a grid reference that is a bare `GridBatch` (not only a VDBTensor) via `_ref_grid`.
- `jcat`/`cat` batch VDBTensors (grids+data) and re-wrap the raw C++ JaggedTensor result.

### transformers version dance
- diffsynth (Wan video) needs `PretrainedConfig` in `transformers.modeling_utils` → re-export shim in `infinicube/__init__.py` (works on transformers 5.x).
- GSM depth encoder needs `transformers.utils.backbone_utils.load_backbone` (removed in 5.x) → **downgraded transformers to 4.49.0**. Video is already generated by then, so diffsynth is no longer needed.

### InfiniCube bug fixed
- `generate_guidance_buffer_trajectory()` was missing the `use_wan_1pt3b` parameter (passed by `main()` but not in the signature nor forwarded) — added.

### Data synthesised for the Autoware clip (not produced by the map converter)
- `data/intrinsic/kashiwanoha.tar` (`intrinsic.front.npy = [fx fy cx cy w h]`, Waymo-front-like).
- Per-frame empty `static_object_info` / `dynamic_object_info` (the buffer step indexes them per frame).
- Trajectory shortened to the first 49 poses (a coherent ~24m segment) near voxel-world 0; Step 2 run with `--extrap_voxel_time 0`. **TODO: have `autoware_hdmap_to_wds.py` emit intrinsic + per-frame object info + a sane trajectory length.**

### gsplat
- Rebuilt v1.4.0 from source for Blackwell (CUDA 12.8, `TORCH_CUDA_ARCH_LIST=12.0+PTX`, torch-lib LDFLAGS). Renders on sm_120.

## Fly-through video "bowl" — root cause & fix

**Symptom:** `kashiwanoha_3dgs_render.mp4` (rendered along the raw ego trajectory, poses 0→48)
showed the scene fanned into a "bowl"/dome from an off-axis-looking view, while the still
`kashiwanoha_3dgs_render.jpg` (a crop of the GSM's own `static_pd_images.jpg`) looked correct.

**Root cause (verified, not a code bug):** rendering the decoded Gaussians with the GSM's own
`render_gsplat_func` at **pose 0** produces the *identical* bowl; at **pose 48** it renders
clean/photorealistic (that pose IS the "good still"). So the camera convention, fov, and
gaussian/pose frame are all correct and match `standard_3dgs_rendering_func`. The feed-forward
GSM (config `view1`, one forward pass) reconstructs a **forward "cone"**: geometry is only dense
and clean when the camera is deep inside it. Sweeping forward-extrapolated cameras shows the
clean, populated region is only about **x ≈ 20 → 34 m** (mean brightness 110→94; by x≈44 it goes
dark/empty). Viewed from the trajectory start (x=0) the camera stares at the poorly-reconstructed
far-field fan → the bowl. The original video simply started at x=0.

**Fix (`scratchpad_render_video.py --dolly X0 X1`):** render a forward dolly that stays inside the
well-reconstructed cone — drive along +x from x=20 to x=30 using the last (clean) pose's
orientation. Result: a smooth 120-frame / 30 fps / 4 s forward-driving view that matches the still
throughout (verified frames 0/60/119). `visualization/kashiwanoha_3dgs_render.mp4`.
Raw trajectory / interpolation modes are still available (`--pose_start/--pose_end`).
The limited clean range is an inherent property of the single feed-forward reconstruction, not the
renderer.
