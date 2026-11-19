"""Resample recorded ForceGello demonstrations without changing the source Zarr."""

import math

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from ImplicitRDP.common.replay_buffer import ReplayBuffer


class IndexedArray:
    """Read a lower-rate phase view from the original Zarr without copying images."""

    def __init__(self, source, indices):
        self.source = source
        self.indices = indices
        self.shape = (len(indices),) + source.shape[1:]
        self.dtype = source.dtype

    def __getitem__(self, item):
        if isinstance(item, tuple):
            rows, *rest = item
            mapped = self.indices[rows]
            if np.isscalar(mapped):
                return self.source[int(mapped)][tuple(rest)]
            if len(mapped) == 0:
                return np.empty((0,) + self.shape[1:], dtype=self.dtype)[(slice(None), *rest)]
            result = self.source.oindex[mapped]
            return result[(slice(None), *rest)]
        rows = self.indices[item]
        if np.isscalar(rows):
            return self.source[int(rows)]
        if len(rows) == 0:
            return np.empty((0,) + self.shape[1:], dtype=self.dtype)
        return self.source.oindex[rows]


def tcp_delta(observed_pose, target_pose, target_gripper):
    """Base-frame TCP displacement, XYZ fixed-axis Euler angles, absolute gripper."""
    observed_pose = np.asarray(observed_pose, dtype=np.float64)
    target_pose = np.asarray(target_pose, dtype=np.float64)
    target_gripper = np.asarray(target_gripper, dtype=np.float64)
    if observed_pose.shape[-1] != 9 or target_pose.shape != observed_pose.shape:
        raise ValueError("TCP delta requires matching 9D observed and target poses")
    if target_gripper.shape != observed_pose.shape[:-1]:
        raise ValueError("TCP delta requires one gripper target per pose")
    if not all(np.all(np.isfinite(x)) for x in (observed_pose, target_pose, target_gripper)):
        raise ValueError("TCP delta inputs must be finite")

    def rotation(pose):
        x = pose[..., 3:6]
        y = pose[..., 6:9]
        z = np.cross(x, y)
        return np.stack((x, y, z), axis=-1)

    r_obs = rotation(observed_pose)
    r_target = rotation(target_pose)
    relative = r_target @ np.swapaxes(r_obs, -1, -2)
    euler_xyz = Rotation.from_matrix(relative).as_euler('xyz')
    return np.concatenate(
        (target_pose[..., :3] - observed_pose[..., :3], euler_xyz,
         target_gripper[..., None]), axis=-1
    ).astype(np.float32)


class TCPDeltaArray:
    """Lazy 7D labels from follower observation[t] and submitted target[t+stride]."""

    def __init__(self, source, side, observed_indices, target_indices):
        self.pose = source[f"{side}_robot_tcp_pose"]
        self.target = source[f"{side}_robot_tcp_command"]
        self.command = source[f"{side}_robot_command"]
        self.observed_indices = observed_indices
        self.target_indices = target_indices
        self.shape = (len(observed_indices), 7)
        self.dtype = np.dtype("float32")

    def __getitem__(self, item):
        if isinstance(item, tuple):
            rows, *rest = item
        else:
            rows, rest = item, []
        obs_rows = self.observed_indices[rows]
        target_rows = self.target_indices[rows]
        if np.isscalar(obs_rows):
            result = tcp_delta(
                self.pose[int(obs_rows)], self.target[int(target_rows)],
                self.command[int(target_rows), 6],
            )
        elif len(obs_rows) == 0:
            result = np.empty((0, 7), dtype=np.float32)
        else:
            result = tcp_delta(
                self.pose.oindex[obs_rows], self.target.oindex[target_rows],
                self.command.oindex[target_rows][:, 6],
            )
        return result[(..., *rest)] if rest else result


def load_joint_replay(zarr_path, keys, action_key, sample_hz, sample_phase=None):
    source = zarr.open_group(zarr_path, mode="r")
    source_hz = float(source.attrs["sample_hz"])
    if not all(math.isfinite(rate) and rate > 0 for rate in (source_hz, sample_hz)):
        raise ValueError("source and training sample rates must be finite and positive")
    ratio = source_hz / sample_hz
    if ratio < 1 or not math.isclose(ratio, round(ratio)):
        raise ValueError(
            "source sample_hz must be an integer multiple of training sample_hz"
        )
    stride = round(ratio)
    if sample_phase is not None and (
        isinstance(sample_phase, bool) or not isinstance(sample_phase, int)
        or not 0 <= sample_phase < stride
    ):
        raise ValueError(f"sample_phase must be an integer in [0, {stride})")
    tcp_action = action_key.endswith("_robot_tcp_delta")
    joint_action = action_key.endswith("_robot_command")
    if tcp_action or joint_action:
        side = action_key.removesuffix("_robot_tcp_delta" if tcp_action else "_robot_command")
        if side not in ("left", "right"):
            raise ValueError("ForceGello action must name the left or right arm")
        valid = source["data"][f"{side}_robot_command_valid"]
    else:
        raise ValueError("ForceGello action must be a robot command or TCP delta")
    # Each source episode supplies one stream per downsampling phase. Keep phases
    # as separate episodes so no training window crosses a phase boundary.
    selected, lengths, groups = [], [], []
    start = 0
    for group, end in enumerate(source["meta/episode_ends"][:]):
        for phase in (range(stride) if sample_phase is None else (sample_phase,)):
            indices = np.arange(start + phase, int(end) - stride, stride, dtype=np.int64)
            if len(indices):
                # Keep only consecutive valid targets; never train through idle gaps.
                valid_rows = np.asarray(valid.oindex[indices + stride], dtype=bool)
                for run in np.split(indices[valid_rows], np.flatnonzero(
                    np.diff(indices[valid_rows]) != stride
                ) + 1):
                    if len(run):
                        selected.append(run)
                        lengths.append(len(run))
                        groups.append(group)
        start = int(end)
    if not selected:
        raise ValueError(
            "no episode contains a valid future command at the training rate"
        )
    indices = np.concatenate(selected)
    data = {key: IndexedArray(source["data"][key], indices)
            for key in keys if key != action_key}
    if tcp_action:
        data[action_key] = TCPDeltaArray(source["data"], side, indices, indices + stride)
    else:
        # Raw command[t] is concurrent with observation[t]. The training action
        # label is the command one training step later, never from another episode.
        data[action_key] = IndexedArray(source["data"][action_key], indices + stride)
    replay = ReplayBuffer(
        {
            "data": data,
            "meta": {"episode_ends": np.cumsum(lengths, dtype=np.int64)},
        }
    )
    replay.source_episode_ids = np.asarray(groups, dtype=np.int64)
    return replay
