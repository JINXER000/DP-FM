"""EE-frame proprio and action utilities."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


EE_LOCAL_PROPRIO = "ee_local_current_v1"


def ee_local_proprio(proprio: np.ndarray) -> np.ndarray:
    """Express an absolute EEF history in its latest EEF frame.

    Each input row is ``xyz + rotation_matrix + gripper_qpos``.  Pose rows are
    re-anchored to the final row while gripper positions remain unchanged.
    """
    values = np.asarray(proprio)
    if (
        values.ndim != 2
        or values.shape[1] != 14
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
    ):
        raise ValueError("proprio must be finite floating-point [To,14]")
    position = values[:, :3].astype(np.float64)
    rotation = values[:, 3:12].reshape(-1, 3, 3).astype(np.float64)
    anchor_position = position[-1]
    anchor_rotation = rotation[-1]
    local_position = (position - anchor_position) @ anchor_rotation
    local_rotation = np.einsum("ij,njk->nik", anchor_rotation.T, rotation)
    result = np.concatenate(
        (
            local_position,
            local_rotation.reshape(len(values), 9),
            values[:, 12:14],
        ),
        axis=-1,
    )
    return result.astype(values.dtype, copy=False)


def bound_pose_action(action: np.ndarray) -> np.ndarray:
    """Project translation and rotation independently onto their L2 unit balls."""
    values = np.asarray(action)
    if values.shape[-1:] != (6,) or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("pose action must end in six floating-point channels")
    result = values.copy()
    for start in (0, 3):
        group = result[..., start : start + 3]
        result[..., start : start + 3] = group / np.maximum(
            1.0, np.linalg.norm(group, axis=-1, keepdims=True)
        )
    return result


def _normalized_pose(pose: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape[-1:] != (7,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite xyz and quaternion xyzw")
    norm = np.linalg.norm(value[..., 3:], axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError(f"{name} contains a zero quaternion")
    result = value.copy()
    result[..., 3:] /= norm
    return result


def ee_action_to_base(action: np.ndarray, eef_pose: np.ndarray) -> np.ndarray:
    """Rotate current-EEF pose increments back to controller base axes."""
    values = np.asarray(action)
    if values.shape[-1:] != (6,) or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("action must end in six floating-point channels")
    pose = _normalized_pose(eef_pose, "eef_pose")
    if pose.shape != (7,):
        raise ValueError("eef_pose must be [7]")
    rotation = Rotation.from_quat(pose[3:]).as_matrix()
    result = values.astype(np.float64, copy=True)
    result[..., :3] = result[..., :3] @ rotation.T
    result[..., 3:] = result[..., 3:] @ rotation.T
    return result.astype(values.dtype, copy=False)

