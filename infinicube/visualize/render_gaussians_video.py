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

"""Render a static 3D Gaussian scene along a (smoothed) camera path to an mp4.

Loads the decoded Gaussians produced by ``scene_gaussian_generation.py``
(``decoded_gs_static.pkl``) and rasterizes them with InfiniCube's own gsplat
backend (:func:`standard_3dgs_rendering_func`) from every pose of a camera
trajectory, then encodes the frames to a video. The trajectory is the clip's ego
poses, optionally interpolated (translation lerp + quaternion slerp) to a denser,
smoother path so the camera "drives through" the scene.

Example
-------
::

    python infinicube/visualize/render_gaussians_video.py \
        --gs_dir  visualization/infinicube_inference/gaussian_scene_generation/trajectory_pose_sample_1frame/kashiwanoha \
        --buf_dir visualization/infinicube_inference/guidance_buffer_generation/trajectory_pose_sample_1frame/kashiwanoha \
        --frames 120 --fps 30 --use_sky \
        --out visualization/kashiwanoha_3dgs_render.mp4

Notes / known issues
--------------------
* This exercises the fvdb / Blackwell port: importing ``infinicube`` installs the
  fvdb 0.2.0->0.3.0 compatibility shim. It needs the runtime env (venv + the
  ``LD_LIBRARY_PATH`` torch-lib / CUDA-12.8 entries — see ``SHIM_PORT_NOTES.md``).
* **Camera convention.** The poses in ``pose.front.npy`` are consumed here as
  camera-to-world (c2w) matrices *as stored*. InfiniCube stores poses in the OpenCV
  convention; ``standard_3dgs_rendering_func`` must receive c2w in the convention it
  expects. If the rendered scene looks warped / "fanned into a bowl" from off-axis
  views, the first thing to check is a convention mismatch (OpenCV vs FLU/OpenGL)
  or an extra world<->grid transform applied to the Gaussians vs. the poses — the
  GSM feed-forward reconstruction is only well-defined near the driven trajectory,
  so novel views away from it legitimately show incomplete geometry too.
"""

import argparse
import os
import pickle

import imageio.v2 as imageio
import numpy as np
import torch

import infinicube  # noqa: F401  (installs the fvdb shim + torch.load patch on import)
from infinicube.utils.gaussian_render_utils import RGB2SH, standard_3dgs_rendering_func
from infinicube.utils.wds_utils import get_sample


def _slerp(q0, q1, t):
    """Spherical linear interpolation between two quaternions (same layout)."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0:  # take the shorter arc
        q1, dot = -q1, -dot
    if dot > 0.9995:  # nearly colinear -> linear interpolation
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(dot) * t
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def _mat_to_quat(R):
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_quat()  # xyzw


def _quat_to_mat(q):
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(q).as_matrix()


def interpolate_poses(poses, n_out):
    """Smoothly resample a (K, 4, 4) c2w trajectory to ``n_out`` poses.

    Translations are linearly interpolated; rotations use quaternion slerp.
    """
    poses = np.asarray(poses, dtype=np.float64)
    k = len(poses)
    if n_out <= k:
        return poses
    ts_in = np.linspace(0.0, 1.0, k)
    ts_out = np.linspace(0.0, 1.0, n_out)
    trans = poses[:, :3, 3]
    quats = np.stack([_mat_to_quat(poses[i, :3, :3]) for i in range(k)])
    out = np.tile(np.eye(4), (n_out, 1, 1))
    for i, t in enumerate(ts_out):
        j = int(np.clip(np.searchsorted(ts_in, t) - 1, 0, k - 2))
        local = (t - ts_in[j]) / (ts_in[j + 1] - ts_in[j] + 1e-9)
        out[i, :3, 3] = trans[j] * (1 - local) + trans[j + 1] * local
        out[i, :3, :3] = _quat_to_mat(_slerp(quats[j], quats[j + 1], local))
    return out


def load_gaussians(gs_dir, device="cuda"):
    """Load ``decoded_gs_static.pkl`` into the dict the renderer expects."""
    with open(os.path.join(gs_dir, "decoded_gs_static.pkl"), "rb") as f:
        gs = pickle.load(f)
    return {
        "xyz": torch.tensor(gs["xyz"], device=device),
        "opacity": torch.tensor(gs["opacity"], device=device),
        "scaling": torch.tensor(gs["scaling"], device=device),
        "rotation": torch.tensor(gs["rotation"], device=device),
        "features": RGB2SH(torch.tensor(gs["rgbs"], device=device)).reshape(-1, 1, 3),
        "sh_degree": 0,
    }


def load_camera(buf_dir):
    """Return (poses c2w (K,4,4), intrinsics as (fx,fy,cx,cy,W,H))."""
    pose = get_sample(os.path.join(buf_dir, "pose.tar"))
    keys = sorted(k for k in pose if "pose.front.npy" in k)
    poses = np.stack([pose[k] for k in keys]).astype(np.float64)
    intr = get_sample(os.path.join(buf_dir, "intrinsic.tar"))["intrinsic.front.npy"]
    fx, fy, cx, cy, w, h = [float(x) for x in intr]
    return poses, (fx, fy, cx, cy, int(w), int(h))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gs_dir", required=True, help="Folder with decoded_gs_static.pkl")
    ap.add_argument("--buf_dir", required=True, help="Folder with pose.tar + intrinsic.tar")
    ap.add_argument("--frames", type=int, default=120, help="Number of output frames (interpolated).")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="visualization/gaussians_render.mp4")
    ap.add_argument("--frames_dir", default=None, help="Optional dir to also dump per-frame jpgs.")
    ap.add_argument("--use_sky", action="store_true", help="Composite the learned skybox behind the scene.")
    args = ap.parse_args()

    device = "cuda"
    gaussians = load_gaussians(args.gs_dir, device)
    print(f"loaded {gaussians['xyz'].shape[0]} gaussians")

    poses, (fx, fy, cx, cy, w, h) = load_camera(args.buf_dir)
    hfov = 2 * np.arctan(w / (2 * fx))
    vfov = 2 * np.arctan(h / (2 * fy))
    print(f"poses={len(poses)} res={w}x{h} hfov={np.degrees(hfov):.1f} vfov={np.degrees(vfov):.1f}")

    skybox_dict = None
    if args.use_sky:
        try:
            from infinicube.utils.sky_utils import read_skybox

            skybox_dict = read_skybox(os.path.join(args.gs_dir, "decoded_gs_static.pkl"))
            print("sky enabled")
        except Exception as exc:  # pragma: no cover
            print("sky disabled (build failed):", repr(exc)[:160])

    traj = interpolate_poses(poses, args.frames)
    print(f"interpolated to {len(traj)} frames")

    if args.frames_dir:
        os.makedirs(args.frames_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    frames = []
    for i, c2w in enumerate(traj):
        img = standard_3dgs_rendering_func(
            c2w, h, w, vfov, hfov, gaussians, scale_modifier=1, skybox_dict=skybox_dict
        )  # uint8 [H, W, 3]
        frames.append(img)
        if args.frames_dir:
            imageio.imwrite(os.path.join(args.frames_dir, f"{i:04d}.jpg"), img)
        if i % 20 == 0:
            print(f"  frame {i}/{len(traj)}", flush=True)

    imageio.mimwrite(
        args.out, frames, fps=args.fps, codec="libx264", quality=8,
        macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    print(f"WROTE {args.out} ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
