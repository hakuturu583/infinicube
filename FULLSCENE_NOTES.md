# Full-scene 3DGS from an Autoware clip

A single voxel world is one ~51.2 m grid crop, so running Step 2/3 on one chunk only
reconstructs that local forward cone (why the earlier single-chunk fly-through had a
short clean range and needed `--dolly`). The full scene needs **every** voxel world
along the trajectory processed and the Gaussians accumulated.

`infinicube/inference/full_scene_from_autoware.py` orchestrates this. It never imports
torch and issues no CUDA itself — the GPU stages run as subprocesses of the existing
scripts, and the accumulation (concat + optional voxel dedupe) is numpy/pickle on CPU.

## Coordinate frame (the accumulation caveat)

`voxel_world_generation.py` (trajectory mode, lines ~827–832) transforms **every** voxel
world `{t}.pt` into the **first camera's FLU frame** and accumulates them:

```python
current_grid_to_first_camera_flu = (
    torch.inverse(self.camera_trajectory_key_poses_flu[0])   # index 0 for ALL steps
    @ self.grid_coord_poses_flu[step]
)
self._update_scene_grid(grid, semantics, current_grid_to_first_camera_flu)
torch.save({"points": self.scene_grid, "semantics": self.scene_semantic}, f"{step}.pt")
```

Consequences:
- **All `{t}.pt` share ONE common frame** (first camera = pose 0) and are *cumulative*
  (`{N-1}.pt` is the whole accumulated map).
- `scene_gaussian_generation.py` decodes static Gaussians in that grid frame, so **every
  chunk's Gaussians are already in the same world frame** → accumulation is a plain
  `np.concatenate`, **no per-chunk transform required**. (This is also why the
  single-chunk render worked directly with the raw `pose.tar` poses: pose 0 is at the
  origin, so world ≈ first-camera frame.)
- If a future GSM variant emitted Gaussians in a *local* per-chunk frame instead, you
  would have to left-multiply each chunk's xyz by that chunk's `grid→first_camera`
  transform before concatenating. It does not here.

`--dedupe_voxel S` optionally keeps one Gaussian per `S`-metre voxel to trim overlap
between consecutive cumulative chunks (they re-decode shared voxels).

## Ready-to-run (post-training, on GPU)

```bash
# 0) env (the libc10 landmine — see SHIM_PORT_NOTES.md)
source scratchpad_env.sh    # venv + LD_LIBRARY_PATH=<venvtorch>/lib:/home/kataoka/cuda-12.8/lib64

# 1) regenerate the map-scale clip (CPU) — full-route trajectory + all tars
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python infinicube/data_process/autoware_hdmap_to_wds.py \
    --osm sample_maps/kashiwanoha_lanelet2_map.osm --clip kashiwanoha --output_root data

# 2) Step 1 over the full trajectory -> N voxel worlds {t}.pt  (GPU)
python infinicube/inference/voxel_world_generation.py none --mode trajectory \
    --use_ema --use_ddim --ddim_step 100 \
    --local_config infinicube/voxelgen/configs/diffusion_64x64x64_dense_vs02_map_cond.yaml \
    --local_checkpoint_path checkpoints/voxel_diffusion.ckpt \
    --clip kashiwanoha --webdataset_root data --target_pose_num 8

# 3) full-scene orchestration: loop Steps 2/3 per chunk, accumulate, render  (GPU)
python infinicube/inference/full_scene_from_autoware.py \
    --clip kashiwanoha \
    --extrap_voxel_root visualization/infinicube_inference/voxel_world_generation/trajectory \
    --data_root data \
    --gsm_config infinicube/voxelgen/configs/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.yaml \
    --gsm_ckpt checkpoints/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.ckpt \
    --video_ckpt checkpoints/wan1pt3b-t2v-buffer-step-3500.safetensors --use_wan_1pt3b \
    --use_sky --frames 240 --fps 30 \
    --out_pkl visualization/kashiwanoha_fullscene/decoded_gs_static.pkl \
    --out_mp4 visualization/kashiwanoha_fullscene_render.mp4
```

## Dry run (CPU only, no GPU — safe while the GPU is busy)

Append `--dry-run` to step 3. It discovers the `{t}.pt` voxel worlds, validates every
input (data tars, GSM config/ckpt, video ckpt) and prints the exact per-chunk commands +
the accumulate/render plan, doing **zero** GPU work. Useful flags: `--chunks 0,3,6`
(process a subset), `--dedupe_voxel 0.1`, `--frames/--fps`.

## Notes / gotchas
- Step 1 must run first (produces the `{t}.pt`). `target_pose_num`/`pose_distance_interval`
  set how many voxel worlds N you get along the route.
- The final fly-through uses the real `data/pose.tar` trajectory (no `--dolly` needed once
  coverage is full). `--dolly` remains in `render_gaussians_video.py` for single-chunk use.
- Per-chunk intermediates go under `--workdir` (`.../fullscene/chunk_{t}/{buffers,gaussians}`);
  the combined scene is `--out_pkl`. One chunk's sky token/modulator is copied next to the
  combined pkl so `--use_sky` works on the merged scene.

## Resident models (no per-chunk reload)

Naively looping Steps 2/3 as a subprocess per chunk reloads the heavy models every
chunk: **T5 umt5-xxl (11 GB) + Wan1.3B** for Step 2 and the **GSM** for Step 3 — minutes
of pure load per chunk. The orchestrator now defaults to a **RESIDENT two-pass** design
that loads each model once and reuses it across all N chunks:

- **Pass A** (video model resident): `guidance_buffer_generation.run_guidance_buffer_for_chunk(...)`
  per chunk. The Wan pipeline is lazily built and cached on
  `generate_guidance_buffer_and_save._video_generator`, so it loads only on chunk 0 and
  is reused for chunks 1..N-1.
- **Pass B** (GSM resident): `scene_gaussian_generation.build_gsm_model()` **once**, then
  `run_gsm_for_folder(model, data_folder, output_folder)` per chunk.

Seams extracted (original CLIs preserved — each `main()` now just calls build-once then
run-one):
- `scene_gaussian_generation.py`: `build_gsm_model(cli_args)` → `(net_model_gsm, args)`;
  `run_gsm_for_folder(net_model_gsm, args, data_folder, output_folder)`.
- `guidance_buffer_generation.py`: `run_guidance_buffer_for_chunk(clip, extrap_voxel_time,
  extrap_voxel_root, output_root, ...)` (thin wrapper; residency via the existing cache).

`--subprocess` restores the old per-chunk-subprocess behavior as a fallback. The
orchestrator imports torch only lazily inside the resident passes, so `--dry-run` and
module import stay CPU/torch-free.

**Expected speedup:** the per-chunk model load (tens of seconds to minutes each,
×2 models ×N chunks) is paid **once** instead of N times. With residency the per-chunk
cost is just the actual work — video generation (~1.5 s/step × ~50 steps ≈ 75 s) + GSM
decode (~1–2 s) per chunk — plus one T5+Wan load and one GSM load for the whole run.
**Real wall-clock timing must be measured post-training** (no GPU runs were performed
for this refactor).
