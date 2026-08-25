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

"""Convert an Autoware Lanelet2 HD map (``.osm``) into the webdataset layout that
InfiniCube's map-conditioned voxel world generation consumes.

InfiniCube conditions the voxel diffusion model on three map point clouds
(``road_edge``, ``road_line``, ``road_surface``) plus an ego trajectory. In the
original pipeline these come from the Waymo Open Dataset. This module produces the
exact same webdataset tar files from an Autoware Lanelet2 map, so the pretrained
InfiniCube checkpoints can be driven by real Autoware HD maps.

The lanelet2 map is read through the ``simple_lanelet2`` package
(https://github.com/hakuturu583/simple_lanelet2), a pip-installable, drop-in
Lanelet2 Python API that needs no C++/Boost toolchain.

Mapping from Lanelet2 to InfiniCube map primitives
--------------------------------------------------
* ``road_surface`` : the drivable area, densely sampled between the left and right
  bound of every drivable lanelet (subtype ``road`` / ``road_shoulder``).
* ``road_line``    : interior lane dividers, i.e. bounds shared by two drivable
  lanelets, plus standalone painted markings (``line_thin`` / ``line_thick`` /
  ``stop_line`` / ``pedestrian_marking``).
* ``road_edge``    : the outer extent of the drivable region, i.e. bounds used by a
  single drivable lanelet, plus physical edges (``road_border`` / ``curbstone`` /
  ``guard_rail`` / ``fence`` / ``wall``) and ``virtual`` boundaries.

The discretization / road-surface estimation reuses the very same helpers as the
Waymo pipeline (:func:`polylines_to_discrete_points`,
:func:`estimate_road_surface_in_world`), so the output is format-identical.

Example
-------
::

    python infinicube/data_process/autoware_hdmap_to_wds.py \
        --osm sample_maps/lanelet2_map.osm \
        --clip my_autoware_scene \
        --output_root data \
        --target_pose_num 8

The produced clip can then be fed to voxel world generation::

    python infinicube/inference/voxel_world_generation.py none \
        --mode trajectory --use_ema --use_ddim --ddim_step 100 \
        --local_config infinicube/voxelgen/configs/diffusion_64x64x64_dense_vs02_map_cond.yaml \
        --local_checkpoint_path checkpoints/voxel_diffusion.ckpt \
        --clip my_autoware_scene --webdataset_root data --target_pose_num 8
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from loguru import logger

from infinicube.data_process.utils import (
    estimate_road_surface_in_world,
    polylines_to_discrete_points,
)
from infinicube.utils.wds_utils import write_to_tar

# --- Lanelet2 subtype / type vocabulary (Autoware conventions) ----------------

# lanelet (relation) subtypes that represent drivable surface
DRIVABLE_LANELET_SUBTYPES = {"road", "road_shoulder", "highway", "play_street"}

# linestring types that are physical road edges (curb / barrier)
PHYSICAL_EDGE_TYPES = {
    "road_border",
    "curbstone",
    "guard_rail",
    "guardrail",
    "fence",
    "wall",
    "road_shoulder",
}

# linestring types that are painted lane markings
MARKING_TYPES = {"line_thin", "line_thick", "stop_line", "pedestrian_marking"}

# linestring types that mark road extent without a physical marking
VIRTUAL_EDGE_TYPES = {"virtual", "road_border_virtual"}


# --- Coordinate handling ------------------------------------------------------


def _read_origin_latlon(osm_path):
    """Return the (lat, lon) of the first node, used as the projection origin."""
    for _, elem in ET.iterparse(osm_path, events=("start",)):
        if elem.tag == "node":
            lat, lon = elem.get("lat"), elem.get("lon")
            if lat is not None and lon is not None:
                return float(lat), float(lon)
            break
    return None


def _load_map_with_local_frame(osm_path):
    """Load a lanelet2 map and return it together with a callable that maps a
    lanelet2 ``Point3d`` to a local metric ``(x, y, z)`` in world coordinates.

    Autoware maps carry both ``lat``/``lon`` and MGRS ``local_x``/``local_y`` tags.
    We prefer the map's own ``local_x``/``local_y``/``ele`` frame when present (it is
    already a consistent planar metric frame). Otherwise we fall back to a UTM
    projection centered on the first node, which the ``simple_lanelet2`` library
    computes for us.
    """
    import lanelet2
    from lanelet2.io import Origin, loadRobust
    from lanelet2.projection import UtmProjector

    origin = _read_origin_latlon(osm_path)
    if origin is None:
        raise ValueError(f"No geo-referenced node found in {osm_path}")

    projector = UtmProjector(Origin(*origin))
    # loadRobust tolerates Autoware-specific regulatory elements (detection_area,
    # no_stopping_area, ...) that the vanilla parser does not know about.
    result = loadRobust(str(osm_path), projector)
    lanelet_map, errors = result if isinstance(result, tuple) else (result, [])
    if errors:
        logger.warning(
            f"lanelet2 reported {len(errors)} non-fatal parse issue(s) "
            f"(usually unknown regulatory elements); geometry is unaffected."
        )

    # Read the raw MGRS local_x/local_y/ele tags per node id, if available.
    local_xyz = _read_local_xyz(osm_path)

    if local_xyz:
        logger.info("Using Autoware MGRS local_x/local_y/ele node frame.")

        def to_xyz(point):
            xyz = local_xyz.get(point.id)
            if xyz is not None:
                return xyz
            return (point.x, point.y, point.z)
    else:
        logger.info("Using UTM projection centered on the map origin.")

        def to_xyz(point):
            return (point.x, point.y, point.z)

    return lanelet_map, to_xyz, lanelet2


def _read_local_xyz(osm_path):
    """Return {node_id: (local_x, local_y, ele)} from the raw OSM, if tagged."""
    local_xyz = {}
    for _, elem in ET.iterparse(osm_path, events=("end",)):
        if elem.tag != "node":
            continue
        tags = {t.get("k"): t.get("v") for t in elem.findall("tag")}
        if "local_x" in tags and "local_y" in tags:
            local_xyz[int(elem.get("id"))] = (
                float(tags["local_x"]),
                float(tags["local_y"]),
                float(tags.get("ele", 0.0)),
            )
        elem.clear()
    return local_xyz


# --- Geometry extraction ------------------------------------------------------


def _linestring_to_polyline(linestring, to_xyz):
    return [list(to_xyz(pt)) for pt in linestring]


def extract_map_polylines(lanelet_map, to_xyz):
    """Classify every relevant linestring into road_edge / road_line polylines and
    collect drivable-lanelet centerlines (``lane``) used for surface estimation.

    Returns:
        dict with keys 'road_edge', 'road_line', 'lane', each a list of polylines
        (a polyline is a list of ``[x, y, z]``), and 'lanelets': list of
        (left_polyline, right_polyline) for drivable lanelets.
    """
    # Count how many drivable lanelets reference each bound linestring id.
    bound_ref_count = {}
    drivable = []
    for lanelet in lanelet_map.laneletLayer:
        subtype = dict(lanelet.attributes).get("subtype")
        if subtype not in DRIVABLE_LANELET_SUBTYPES:
            continue
        drivable.append(lanelet)
        for bound in (lanelet.leftBound, lanelet.rightBound):
            bound_ref_count[bound.id] = bound_ref_count.get(bound.id, 0) + 1

    road_edge, road_line, lane = [], [], []
    lanelets_lr = []
    classified_bound_ids = set()

    for lanelet in drivable:
        left = _linestring_to_polyline(lanelet.leftBound, to_xyz)
        right = _linestring_to_polyline(lanelet.rightBound, to_xyz)
        lanelets_lr.append((left, right))
        # centerline proxy for road-surface estimation (matches Waymo "lane")
        lane.append(_lanelet_centerline(left, right))

        for bound in (lanelet.leftBound, lanelet.rightBound):
            if bound.id in classified_bound_ids:
                continue
            classified_bound_ids.add(bound.id)
            btype = dict(bound.attributes).get("type")
            polyline = _linestring_to_polyline(bound, to_xyz)
            if btype in PHYSICAL_EDGE_TYPES or btype in VIRTUAL_EDGE_TYPES:
                road_edge.append(polyline)
            elif bound_ref_count[bound.id] == 1:
                # outer boundary of the drivable region
                road_edge.append(polyline)
            else:
                # interior divider shared by two lanelets
                road_line.append(polyline)

    # Standalone painted markings not already used as a drivable bound.
    for linestring in lanelet_map.lineStringLayer:
        if linestring.id in classified_bound_ids:
            continue
        ltype = dict(linestring.attributes).get("type")
        if ltype in MARKING_TYPES:
            road_line.append(_linestring_to_polyline(linestring, to_xyz))
        elif ltype in PHYSICAL_EDGE_TYPES:
            road_edge.append(_linestring_to_polyline(linestring, to_xyz))

    return {
        "road_edge": road_edge,
        "road_line": road_line,
        "lane": lane,
        "lanelets": lanelets_lr,
    }


def _resample_polyline(polyline, num):
    """Resample a polyline to ``num`` points evenly by arc length."""
    pts = np.asarray(polyline, dtype=np.float64)
    if len(pts) == 1:
        return np.repeat(pts, num, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = cumulative[-1]
    if total < 1e-9:
        return np.repeat(pts[:1], num, axis=0)
    targets = np.linspace(0.0, total, num)
    out = np.empty((num, 3))
    for d in range(3):
        out[:, d] = np.interp(targets, cumulative, pts[:, d])
    return out


def _lanelet_centerline(left, right, num=None):
    """Centerline as the midpoint of arc-length-matched left/right bounds."""
    if num is None:
        num = max(len(left), len(right), 2)
    left_r = _resample_polyline(left, num)
    right_r = _resample_polyline(right, num)
    return ((left_r + right_r) * 0.5).tolist()


def sample_lanelet_surface(lanelets_lr, spacing=0.4):
    """Densely sample the drivable surface between the left/right bound of each
    lanelet at roughly ``spacing`` meters, returning an (N, 3) point cloud."""
    points = []
    for left, right in lanelets_lr:
        left_arr = np.asarray(left, dtype=np.float64)
        right_arr = np.asarray(right, dtype=np.float64)
        # longitudinal resolution from the mean bound length
        length = 0.5 * (
            np.linalg.norm(np.diff(left_arr, axis=0), axis=1).sum()
            + np.linalg.norm(np.diff(right_arr, axis=0), axis=1).sum()
        )
        n_long = max(int(np.ceil(length / spacing)) + 1, 2)
        left_r = _resample_polyline(left, n_long)
        right_r = _resample_polyline(right, n_long)
        # lateral resolution from the local width
        for i in range(n_long):
            width = np.linalg.norm(right_r[i] - left_r[i])
            n_lat = max(int(np.ceil(width / spacing)) + 1, 2)
            s = np.linspace(0.0, 1.0, n_lat)[:, None]
            cross = (1.0 - s) * left_r[i][None, :] + s * right_r[i][None, :]
            points.append(cross)
    if not points:
        return np.zeros((0, 3))
    return np.concatenate(points, axis=0)


def _voxel_downsample(points, voxel_size):
    """Deduplicate points onto a voxel grid (keeps one point per occupied voxel)."""
    if len(points) == 0:
        return points
    keys = np.round(np.asarray(points) / voxel_size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return np.asarray(points)[np.sort(idx)]


# --- Ego trajectory -----------------------------------------------------------


def _flu_to_opencv_matrix(pose_flu):
    """Convert a (4, 4) FLU camera pose to the opencv convention InfiniCube stores.

    FLU axes (x-forward, y-left, z-up) -> opencv axes (x-right, y-down, z-forward).
    This mirrors ``infinicube.camera.base.flu_to_opencv`` but is inlined to keep the
    converter free of the cv2 dependency.
    """
    pose_cv = np.empty_like(pose_flu)
    pose_cv[..., 0] = -pose_flu[..., 1]  # right  = -left
    pose_cv[..., 1] = -pose_flu[..., 2]  # down   = -up
    pose_cv[..., 2] = pose_flu[..., 0]  # forward = forward
    pose_cv[..., 3] = pose_flu[..., 3]
    return pose_cv


def build_ego_trajectory(lanelets_lr, lanelet_map, to_xyz, spacing=0.5, z_offset=0.0):
    """Build a dense sequence of ego poses following the longest connected chain of
    drivable lanelets.

    Returns an (K, 4, 4) array of poses in the opencv convention, expressed in the
    same world frame as the map points.
    """
    centerlines = [
        np.asarray(_lanelet_centerline(left, right), dtype=np.float64)
        for left, right in lanelets_lr
    ]
    if not centerlines:
        raise ValueError("No drivable lanelets found; cannot build a trajectory.")

    chain = _longest_lanelet_chain(centerlines)
    path = np.concatenate([centerlines[i] for i in chain], axis=0)
    path = _dedup_consecutive(path)

    # Resample the path at a fixed spacing.
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total = seg.sum()
    n = max(int(np.ceil(total / spacing)) + 1, 2)
    path = _resample_polyline(path.tolist(), n)
    path[:, 2] += z_offset

    return _poses_from_path(path)


def _dedup_consecutive(path, tol=1e-3):
    keep = [0]
    for i in range(1, len(path)):
        if np.linalg.norm(path[i] - path[keep[-1]]) > tol:
            keep.append(i)
    return path[keep]


def _longest_lanelet_chain(centerlines, tol=1.0):
    """Greedily connect lanelets whose centerline endpoints touch, returning the
    indices of the longest chain found."""
    n = len(centerlines)
    starts = np.array([c[0] for c in centerlines])
    ends = np.array([c[-1] for c in centerlines])

    # successor[i] = j if end of i coincides with start of j
    successor = {}
    for i in range(n):
        d = np.linalg.norm(starts - ends[i], axis=1)
        d[i] = np.inf
        j = int(np.argmin(d))
        if d[j] < tol:
            successor[i] = j

    best = []
    for seed in range(n):
        visited = set()
        chain = []
        node = seed
        while node is not None and node not in visited:
            visited.add(node)
            chain.append(node)
            node = successor.get(node)
        if len(chain) > len(best):
            best = chain
    return best if best else [0]


def _poses_from_path(path):
    """Turn an ordered (K, 3) path into (K, 4, 4) opencv poses.

    The FLU frame at each waypoint has x pointing along the path tangent and z up.
    """
    k = len(path)
    tangents = np.zeros_like(path)
    tangents[:-1] = path[1:] - path[:-1]
    tangents[-1] = tangents[-2] if k > 1 else np.array([1.0, 0.0, 0.0])

    poses = np.zeros((k, 4, 4))
    up = np.array([0.0, 0.0, 1.0])
    for i in range(k):
        forward = tangents[i]
        norm = np.linalg.norm(forward)
        forward = forward / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
        left = np.cross(up, forward)
        left_norm = np.linalg.norm(left)
        left = left / left_norm if left_norm > 1e-9 else np.array([0.0, 1.0, 0.0])
        real_up = np.cross(forward, left)

        pose_flu = np.eye(4)
        pose_flu[:3, 0] = forward
        pose_flu[:3, 1] = left
        pose_flu[:3, 2] = real_up
        pose_flu[:3, 3] = path[i]
        poses[i] = _flu_to_opencv_matrix(pose_flu)
    return poses


# --- Writing webdataset tars --------------------------------------------------


def write_infinicube_wds(
    clip,
    output_root,
    road_edge_pts,
    road_line_pts,
    road_surface_pts,
    ego_poses,
):
    """Write the tar files that :func:`infinicube.voxelgen.utils.extrap_util.get_wds_data`
    expects for map-conditioned voxel world generation."""
    output_root = Path(output_root)

    write_to_tar(
        {"road_edge.npy": road_edge_pts.astype(np.float32)},
        output_root / "3d_road_edge_voxelsize_025" / f"{clip}.tar",
        __key__=clip,
    )
    write_to_tar(
        {"road_line.npy": road_line_pts.astype(np.float32)},
        output_root / "3d_road_line_voxelsize_025" / f"{clip}.tar",
        __key__=clip,
    )
    write_to_tar(
        {"road_surface.npy": road_surface_pts.astype(np.float32)},
        output_root / "3d_road_surface_voxelsize_04" / f"{clip}.tar",
        __key__=clip,
    )

    pose_sample = {"__key__": clip}
    for idx, pose in enumerate(ego_poses):
        pose_sample[f"{idx:06d}.pose.front.npy"] = pose.astype(np.float64)
    write_to_tar(pose_sample, output_root / "pose" / f"{clip}.tar")

    # Empty static-object info (no vehicles baked into the HD map). get_wds_data
    # loads this to build 3D boxes; an empty dict yields zero boxes.
    write_to_tar(
        {"000000.static_object_info.json": {}},
        output_root / "static_object_info" / f"{clip}.tar",
        __key__=clip,
    )


def convert(
    osm_path,
    clip,
    output_root,
    segment_interval=0.25,
    surface_spacing=0.4,
    pose_spacing=0.5,
    ego_z_offset=0.0,
):
    """End-to-end conversion of a single Autoware lanelet2 map to an InfiniCube clip."""
    logger.info(f"Loading Autoware lanelet2 map: {osm_path}")
    lanelet_map, to_xyz, _ = _load_map_with_local_frame(osm_path)
    logger.info(
        f"laneletLayer={len(lanelet_map.laneletLayer)} "
        f"lineStringLayer={len(lanelet_map.lineStringLayer)}"
    )

    polylines = extract_map_polylines(lanelet_map, to_xyz)
    logger.info(
        f"road_edge polylines={len(polylines['road_edge'])} "
        f"road_line polylines={len(polylines['road_line'])} "
        f"drivable lanelets={len(polylines['lanelets'])}"
    )

    # road_edge / road_line: interpolate polylines to discrete points, then
    # voxel-downsample at 0.25 m (matches 3d_road_*_voxelsize_025).
    road_edge_pts = polylines_to_discrete_points(polylines["road_edge"], segment_interval)
    road_line_pts = polylines_to_discrete_points(polylines["road_line"], segment_interval)
    road_edge_pts = _voxel_downsample(road_edge_pts, 0.25)
    road_line_pts = _voxel_downsample(road_line_pts, 0.25)

    # road_surface: sample the drivable area and estimate a clean surface using the
    # same block-plane estimator as the Waymo pipeline, falling back to the raw
    # sampling if the estimator has too little support.
    lane_pts = polylines_to_discrete_points(polylines["lane"], segment_interval)
    surface_raw = sample_lanelet_surface(polylines["lanelets"], surface_spacing)
    try:
        if len(lane_pts) > 0 and len(road_edge_pts) > 0:
            road_surface_pts = estimate_road_surface_in_world(
                road_edge_pts, lane_pts, block_size=[40, 40], voxel_sizes=[0.4, 0.4, 0.2]
            )
        else:
            road_surface_pts = surface_raw
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(f"Road-surface estimator failed ({exc}); using dense sampling.")
        road_surface_pts = surface_raw
    road_surface_pts = _voxel_downsample(road_surface_pts, surface_spacing)

    logger.info(
        f"points -> road_edge={len(road_edge_pts)} "
        f"road_line={len(road_line_pts)} road_surface={len(road_surface_pts)}"
    )

    ego_poses = build_ego_trajectory(
        polylines["lanelets"],
        lanelet_map,
        to_xyz,
        spacing=pose_spacing,
        z_offset=ego_z_offset,
    )
    logger.info(f"ego trajectory: {len(ego_poses)} poses")

    write_infinicube_wds(
        clip,
        output_root,
        road_edge_pts,
        road_line_pts,
        road_surface_pts,
        ego_poses,
    )
    logger.info(f"Done. Wrote clip '{clip}' under {Path(output_root).resolve()}")

    return {
        "road_edge": road_edge_pts,
        "road_line": road_line_pts,
        "road_surface": road_surface_pts,
        "ego_poses": ego_poses,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert an Autoware Lanelet2 HD map to InfiniCube map-condition "
        "webdataset tars."
    )
    parser.add_argument("--osm", required=True, help="Path to the lanelet2 .osm map.")
    parser.add_argument("--clip", required=True, help="Clip name for the output.")
    parser.add_argument(
        "--output_root", default="data", help="Webdataset root (default: data)."
    )
    parser.add_argument(
        "--segment_interval",
        type=float,
        default=0.25,
        help="Polyline discretization interval in meters (default: 0.25).",
    )
    parser.add_argument(
        "--surface_spacing",
        type=float,
        default=0.4,
        help="Road-surface sampling spacing in meters (default: 0.4).",
    )
    parser.add_argument(
        "--pose_spacing",
        type=float,
        default=0.5,
        help="Ego trajectory waypoint spacing in meters (default: 0.5).",
    )
    parser.add_argument(
        "--ego_z_offset",
        type=float,
        default=0.0,
        help="Vertical offset added to ego poses above the lane surface (default: 0).",
    )
    args = parser.parse_args()

    convert(
        osm_path=args.osm,
        clip=args.clip,
        output_root=args.output_root,
        segment_interval=args.segment_interval,
        surface_spacing=args.surface_spacing,
        pose_spacing=args.pose_spacing,
        ego_z_offset=args.ego_z_offset,
    )


if __name__ == "__main__":
    main()
