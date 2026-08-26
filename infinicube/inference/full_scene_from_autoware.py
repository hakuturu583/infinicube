# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Full-scene 3D Gaussian reconstruction from an Autoware HD-map clip.

A single voxel world only covers ~51.2 m (one grid crop), so running Step 2/3 on
one chunk reconstructs just that local forward cone. This orchestrator processes
**every** voxel world produced by Step 1 along the trajectory, runs the guidance-
buffer + video (Step 2) and scene-Gaussian (Step 3) stages for each, then
**accumulates** all chunks' static Gaussians into one scene and renders a
full-trajectory fly-through.

Coordinate frame (verified from ``voxel_world_generation.py`` lines ~827-832):
    Every voxel world ``{t}.pt`` is transformed into the **first camera's FLU
    frame** and accumulated, so all chunks — and therefore all decoded Gaussians,
    which live in that grid frame — share ONE common world frame. Accumulation is a
    plain concatenation; no per-chunk transform is needed. (This is why the
    single-chunk render worked with the raw ``pose.tar`` poses.)

This script never imports torch and issues no CUDA itself: the GPU stages (Steps
2/3 and the final render) are launched as subprocesses of the existing scripts.
``--dry-run`` validates every input/path and prints the full plan with **no GPU
work at all**.

READY-TO-RUN (post-training, on GPU) -- set the env first (the libc10 landmine):
    source scratchpad_env.sh   # venv + LD_LIBRARY_PATH=<venvtorch>/lib:/home/kataoka/cuda-12.8/lib64
    python infinicube/inference/full_scene_from_autoware.py \
        --clip kashiwanoha \
        --extrap_voxel_root visualization/infinicube_inference/voxel_world_generation/trajectory \
        --data_root data \
        --gsm_config infinicube/voxelgen/configs/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.yaml \
        --gsm_ckpt checkpoints/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.ckpt \
        --video_ckpt checkpoints/wan1pt3b-t2v-buffer-step-3500.safetensors --use_wan_1pt3b \
        --workdir visualization/infinicube_inference/fullscene \
        --out_pkl visualization/kashiwanoha_fullscene/decoded_gs_static.pkl \
        --out_mp4 visualization/kashiwanoha_fullscene_render.mp4

    (Step 1 must have been run first to produce the {t}.pt voxel worlds, e.g.
     python infinicube/inference/voxel_world_generation.py none --mode trajectory ...
     --clip kashiwanoha ... — see README.)

DRY RUN (CPU only, no GPU): append ``--dry-run`` to the command above.
"""
import argparse
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# helpers (CPU only)
# --------------------------------------------------------------------------- #
def _log(msg):
    print(f"[full-scene] {msg}", flush=True)


def discover_voxel_worlds(extrap_voxel_root, clip):
    """Return the sorted list of voxel-world .pt files for the clip."""
    d = Path(extrap_voxel_root) / clip
    files = sorted(d.glob("*.pt"), key=lambda p: int(p.stem))
    return files


def check_inputs(args):
    """Validate every path Steps 2/3 + render will need. Returns (ok, problems, voxel_files)."""
    problems = []
    voxel_files = discover_voxel_worlds(args.extrap_voxel_root, args.clip)
    if not voxel_files:
        problems.append(
            f"No voxel worlds {args.extrap_voxel_root}/{args.clip}/*.pt — run Step 1 first."
        )
    data = Path(args.data_root)
    required_tars = [
        data / "pose" / f"{args.clip}.tar",
        data / "intrinsic" / f"{args.clip}.tar",
        data / "static_object_info" / f"{args.clip}.tar",
        data / "dynamic_object_info" / f"{args.clip}.tar",
        data / "3d_road_surface_voxelsize_04" / f"{args.clip}.tar",
    ]
    for t in required_tars:
        if not t.exists():
            problems.append(f"missing data tar: {t}  (regenerate with autoware_hdmap_to_wds.py)")
    for f in [args.gsm_config, args.gsm_ckpt, args.video_ckpt]:
        if not Path(f).exists():
            problems.append(f"missing checkpoint/config: {f}")
    return (len(problems) == 0), problems, voxel_files


def step2_cmd(args, t, chunk_out_root):
    cmd = [
        sys.executable, "infinicube/inference/guidance_buffer_generation.py",
        "--mode", "trajectory", "--clip", args.clip,
        "--extrap_voxel_root", args.extrap_voxel_root,
        "--extrap_voxel_time", str(t),
        "--output_root", str(chunk_out_root),
        "--data_root", args.data_root,
        "--offset_unit", "frame", "--offset", "1",
        "--video_checkpoint_path", args.video_ckpt,
    ]
    if args.use_wan_1pt3b:
        cmd.append("--use_wan_1pt3b")
    return cmd


def step2_output_folder(chunk_out_root, clip):
    # guidance_buffer_generation writes to <output_root>/trajectory_pose_sample_1frame/<clip>
    return Path(chunk_out_root) / "trajectory_pose_sample_1frame" / clip


def step3_cmd(args, data_folder, gs_out_root):
    return [
        sys.executable, "infinicube/inference/scene_gaussian_generation.py", "none",
        "--data_folder", str(data_folder),
        "--local_config", args.gsm_config,
        "--local_checkpoint_path", args.gsm_ckpt,
        "--output_folder", str(gs_out_root),
    ]


def step3_output_folder(gs_out_root, clip):
    # scene_gaussian_generation writes to <output_root>/<...>/<clip>/decoded_gs_static.pkl.
    # We pass a per-chunk output_root so each chunk lands in its own folder; the pkl is
    # located by globbing for decoded_gs_static.pkl underneath it.
    return Path(gs_out_root)


def run(cmd, cwd=REPO):
    _log("RUN: " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def find_pkl(root):
    hits = sorted(Path(root).rglob("decoded_gs_static.pkl"))
    return hits[0] if hits else None


def accumulate(pkls, out_pkl, dedupe_voxel=None):
    """Concatenate chunk Gaussians (shared world frame) into one scene; optional
    voxel-downsample dedupe of overlaps. CPU / numpy only."""
    keys = ["xyz", "opacity", "scaling", "rotation", "rgbs"]
    acc = {k: [] for k in keys}
    total = 0
    for p in pkls:
        with open(p, "rb") as f:
            gs = pickle.load(f)
        n = gs["xyz"].shape[0]
        total += n
        for k in keys:
            acc[k].append(np.asarray(gs[k]))
        _log(f"  + {n:>9,} gaussians from {p}")
    merged = {k: np.concatenate(acc[k], axis=0) for k in keys}
    _log(f"concatenated {total:,} gaussians")

    if dedupe_voxel and dedupe_voxel > 0:
        keys_grid = np.round(merged["xyz"] / dedupe_voxel).astype(np.int64)
        _, idx = np.unique(keys_grid, axis=0, return_index=True)
        idx = np.sort(idx)
        for k in keys:
            merged[k] = merged[k][idx]
        _log(f"voxel-downsampled @ {dedupe_voxel} m -> {len(idx):,} gaussians")

    out_pkl = Path(out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    from collections import OrderedDict
    with open(out_pkl, "wb") as f:
        pickle.dump(OrderedDict((k, merged[k].astype(np.float32)) for k in keys), f)
    _log(f"wrote combined scene -> {out_pkl} ({merged['xyz'].shape[0]:,} gaussians)")
    return out_pkl


def run_subprocess_passes(args, idxs):
    """Fallback: fresh subprocess per chunk (reloads models each chunk). Returns pkls."""
    chunk_pkls = []
    for t in idxs:
        chunk_root = Path(args.workdir) / f"chunk_{t}"
        run(step2_cmd(args, t, chunk_root / "buffers"))
        data_folder = step2_output_folder(chunk_root / "buffers", args.clip)
        run(step3_cmd(args, data_folder, chunk_root / "gaussians"))
        pkl = find_pkl(chunk_root / "gaussians")
        if pkl is None:
            _log(f"WARNING: no decoded_gs_static.pkl for chunk {t}; skipping")
            continue
        chunk_pkls.append(pkl)
    return chunk_pkls


def run_resident_passes(args, idxs):
    """RESIDENT two-pass: load each model ONCE and reuse across all chunks.

    Pass A loads the Wan video pipeline once (cached inside guidance_buffer_generation)
    and generates buffers+video for every chunk. Pass B loads the GSM once and decodes
    Gaussians for every chunk. Imports are lazy so this module stays torch-free until a
    real run. Returns the list of per-chunk decoded_gs_static.pkl paths.
    """
    # ---- Pass A: guidance buffers + video (Wan model resident) ---- #
    _log("Pass A — guidance buffers + video (video model resident across chunks)")
    from infinicube.inference import guidance_buffer_generation as gbg

    for t in idxs:
        chunk_root = Path(args.workdir) / f"chunk_{t}"
        _log(f"[chunk {t}] Pass A: buffers+video -> {chunk_root/'buffers'}")
        gbg.run_guidance_buffer_for_chunk(
            clip=args.clip,
            extrap_voxel_time=t,
            extrap_voxel_root=args.extrap_voxel_root,
            output_root=str(chunk_root / "buffers"),
            data_root=args.data_root,
            video_checkpoint_path=args.video_ckpt,
            use_wan_1pt3b=args.use_wan_1pt3b,
        )

    # ---- Pass B: GSM decode (GSM model resident) ---- #
    _log("Pass B — scene Gaussians (GSM model resident across chunks)")
    from infinicube.inference import scene_gaussian_generation as sgg

    gsm_cli = sgg.get_parser().parse_known_args(
        [
            "--local_config", args.gsm_config,
            "--local_checkpoint_path", args.gsm_ckpt,
            "--output_folder", str(Path(args.workdir) / "chunk_0" / "gaussians"),
        ]
    )[0]
    net_model_gsm, model_args = sgg.build_gsm_model(cli_args=gsm_cli)

    chunk_pkls = []
    for t in idxs:
        chunk_root = Path(args.workdir) / f"chunk_{t}"
        data_folder = step2_output_folder(chunk_root / "buffers", args.clip)
        _log(f"[chunk {t}] Pass B: GSM decode from {data_folder}")
        static_gs_path = sgg.run_gsm_for_folder(
            net_model_gsm, model_args,
            data_folder=str(data_folder),
            output_folder=str(chunk_root / "gaussians"),
        )
        chunk_pkls.append(Path(static_gs_path))
    return chunk_pkls


def render_cmd(args, gs_dir, gs_pkl_name):
    cmd = [
        sys.executable, "infinicube/visualize/render_gaussians_video.py",
        "--gs_dir", str(gs_dir), "--gs_pkl", gs_pkl_name,
        "--buf_dir", str(Path(args.data_root)),  # pose.tar + intrinsic.tar live under data/
        "--frames", str(args.frames), "--fps", str(args.fps),
        "--out", args.out_mp4,
    ]
    if args.use_sky:
        cmd.append("--use_sky")
    return cmd


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--extrap_voxel_root",
                    default="visualization/infinicube_inference/voxel_world_generation/trajectory")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--gsm_config",
                    default="infinicube/voxelgen/configs/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.yaml")
    ap.add_argument("--gsm_ckpt",
                    default="checkpoints/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.ckpt")
    ap.add_argument("--video_ckpt", default="checkpoints/wan1pt3b-t2v-buffer-step-3500.safetensors")
    ap.add_argument("--use_wan_1pt3b", action="store_true",
                    help="Use the Wan2.1-T2V-1.3B video path (matches --video_ckpt).")
    ap.add_argument("--workdir", default="visualization/infinicube_inference/fullscene",
                    help="Per-chunk intermediate outputs (buffers, videos, per-chunk gaussians).")
    ap.add_argument("--out_pkl", default="visualization/kashiwanoha_fullscene/decoded_gs_static.pkl")
    ap.add_argument("--out_mp4", default="visualization/kashiwanoha_fullscene_render.mp4")
    ap.add_argument("--dedupe_voxel", type=float, default=None,
                    help="Optional voxel size (m) to downsample overlapping gaussians in the merge.")
    ap.add_argument("--chunks", type=str, default=None,
                    help="Comma-separated voxel-world indices to process (default: all).")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--use_sky", action="store_true")
    ap.add_argument("--subprocess", action="store_true",
                    help="Fallback: run Steps 2/3 as a fresh subprocess per chunk (reloads models "
                         "every chunk). Default is the faster RESIDENT two-pass mode that loads each "
                         "model once in-process and reuses it across all chunks.")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="Validate inputs, print the plan, do NO GPU work.")
    args = ap.parse_args()

    ok, problems, voxel_files = check_inputs(args)
    idxs = [int(p.stem) for p in voxel_files]
    if args.chunks:
        want = {int(x) for x in args.chunks.split(",")}
        voxel_files = [p for p in voxel_files if int(p.stem) in want]
        idxs = [int(p.stem) for p in voxel_files]

    _log(f"clip={args.clip}  voxel worlds found: {idxs}")
    _log(f"chunks to process: {idxs}")
    combined_dir = Path(args.out_pkl).parent
    combined_name = Path(args.out_pkl).name

    if args.dry_run:
        _log("===== DRY RUN (no GPU) — planned pipeline =====")
        mode = "SUBPROCESS (per-chunk reload)" if args.subprocess else "RESIDENT two-pass (load once)"
        _log(f"execution mode: {mode}")
        if args.subprocess:
            for t in idxs:
                chunk_root = Path(args.workdir) / f"chunk_{t}"
                _log(f"[chunk {t}] Step2: {' '.join(map(str, step2_cmd(args, t, chunk_root / 'buffers')))}")
                df = step2_output_folder(chunk_root / "buffers", args.clip)
                _log(f"[chunk {t}] Step3: {' '.join(map(str, step3_cmd(args, df, chunk_root / 'gaussians')))}")
        else:
            _log("Pass A (video model resident): guidance_buffer_generation.run_guidance_buffer_for_chunk("
                 f"clip={args.clip}, extrap_voxel_time=t, output_root=<workdir>/chunk_t/buffers, "
                 f"use_wan_1pt3b={args.use_wan_1pt3b}) for t in " + str(idxs))
            _log("Pass B (GSM resident): scene_gaussian_generation.build_gsm_model() once, then "
                 "run_gsm_for_folder(model, data_folder=<chunk buffers>/…/clip, "
                 "output_folder=<workdir>/chunk_t/gaussians) for each t")
        _log("Accumulate: concat all chunks' decoded_gs_static.pkl (shared first-camera frame)"
             + (f", dedupe @ {args.dedupe_voxel} m" if args.dedupe_voxel else "")
             + f" -> {args.out_pkl}")
        _log(f"Render: {' '.join(map(str, render_cmd(args, combined_dir, combined_name)))}")
        if not ok:
            _log("PROBLEMS (fix before a real run):")
            for p in problems:
                _log("  - " + p)
            sys.exit(1)
        _log("All inputs present. Dry run OK — remove --dry-run to execute on GPU.")
        return

    if not ok:
        _log("Refusing to run — missing inputs:")
        for p in problems:
            _log("  - " + p)
        sys.exit(1)

    # ---- real run (GPU) ---- #
    if args.subprocess:
        _log("mode: SUBPROCESS (per-chunk model reload)")
        chunk_pkls = run_subprocess_passes(args, idxs)
    else:
        _log("mode: RESIDENT two-pass (each model loaded once)")
        chunk_pkls = run_resident_passes(args, idxs)

    sky_src = chunk_pkls[0] if chunk_pkls else None

    if not chunk_pkls:
        _log("No gaussians produced for any chunk — aborting.")
        sys.exit(1)

    out_pkl = accumulate(chunk_pkls, args.out_pkl, dedupe_voxel=args.dedupe_voxel)

    # Copy one chunk's sky files next to the combined pkl so --use_sky works.
    if args.use_sky and sky_src is not None:
        stem_src = str(sky_src)[: -len("decoded_gs_static.pkl")]
        stem_dst = str(out_pkl)[: -len(Path(out_pkl).name)] + Path(out_pkl).stem
        for suffix in ["_modulator.yaml", "_modulator.pt", "_sky_token.pt"]:
            src = Path(stem_src.rstrip("/") + "/decoded_gs_static" + suffix) \
                if os.path.isdir(stem_src) else Path(stem_src + "decoded_gs_static" + suffix)
            src = Path(str(sky_src).replace("decoded_gs_static.pkl", "decoded_gs_static" + suffix))
            if src.exists():
                shutil.copy(src, stem_dst + suffix)

    run(render_cmd(args, combined_dir, combined_name))
    _log(f"DONE. Full-scene gaussians: {out_pkl}  video: {args.out_mp4}")


if __name__ == "__main__":
    main()
