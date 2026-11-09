"""LIBERO-Spatial image dataset for the follower-compatible task expert."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from diffusion_policy.common.pose_util import (
    EE_LOCAL_PROPRIO,
    bound_pose_action,
    ee_local_proprio,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


DATA_FILE_SUFFIX = "_demo_pc_instance_sg_50.hdf5"
IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
PROPRIO_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
PROPRIO_DIM = 14
ACTION_DIM = 7
RECORDED_CONTROLLER_CONFIG = {
    "type": "OSC_POSE",
    "input_min": -1,
    "input_max": 1,
    "output_min": [-.05, -.05, -.05, -.5, -.5, -.5],
    "output_max": [.05, .05, .05, .5, .5, .5],
    "kp": 150,
    "damping_ratio": 1,
    "impedance_mode": "fixed",
    "uncouple_pos_ori": True,
    "control_delta": True,
    "interpolation": None,
}
RUNTIME_CONTROLLER_REQUIREMENTS = {
    "control_hz": 20,
    "input_ref_frame": "base",
    "goal_update_mode": "achieved",
    "kp": [150.] * 6,
    "kd": [2. * 150.**.5] * 6,
    "controller_origin_orientation": "identity",
}


def _xml_rotation(element: ET.Element) -> np.ndarray:
    """Return a MuJoCo element's local rotation; active data use wxyz quaternions."""
    unsupported = [name for name in ("euler", "axisangle", "xyaxes", "zaxis")
                   if element.get(name) is not None]
    if unsupported:
        raise ValueError(f"unsupported MuJoCo orientation attributes: {unsupported}")
    value = element.get("quat")
    if value is None:
        return np.eye(3, dtype=np.float64)
    wxyz = np.fromstring(value, sep=" ", dtype=np.float64)
    if wxyz.shape != (4,) or not np.isfinite(wxyz).all() or np.linalg.norm(wxyz) < 1e-8:
        raise ValueError(f"invalid MuJoCo quaternion {value!r}")
    xyzw = wxyz[[1, 2, 3, 0]]
    return Rotation.from_quat(xyzw).as_matrix()


def body_to_site_rotation_from_xml(model_xml: str) -> np.ndarray:
    """Recover the fixed Panda hand-body to grip-site rotation from demo XML."""
    root = ET.fromstring(model_xml)
    hands = [element for element in root.iter("body")
             if element.get("name") == "robot0_right_hand"]
    if len(hands) != 1:
        raise ValueError(f"expected one robot0_right_hand body, found {len(hands)}")
    hand = hands[0]
    sites = [element for element in hand.iter("site")
             if (element.get("name") or "").endswith("grip_site")]
    if len(sites) != 1:
        raise ValueError(f"expected one grip_site below robot0_right_hand, found {len(sites)}")
    site = sites[0]
    parents = {child: parent for parent in root.iter() for child in parent}
    chain: list[ET.Element] = []
    current = site
    while current is not hand:
        if current.tag in {"body", "site"}:
            chain.append(current)
        current = parents.get(current)
        if current is None:
            raise ValueError("grip site is not a descendant of robot0_right_hand")
    result = np.eye(3, dtype=np.float64)
    for element in reversed(chain):
        result = result @ _xml_rotation(element)
    return Rotation.from_matrix(result).as_matrix()


def site_rotation_from_body_quaternion(
    body_quaternion_xyzw: np.ndarray,
    body_to_site_rotation: np.ndarray,
) -> np.ndarray:
    body = np.asarray(body_quaternion_xyzw, dtype=np.float64)
    if body.shape[-1:] != (4,) or not np.isfinite(body).all():
        raise ValueError("body quaternion must be finite [...,4] xyzw")
    transform = np.asarray(body_to_site_rotation, dtype=np.float64)
    if transform.shape != (3, 3):
        raise ValueError("body_to_site_rotation must be [3,3]")
    return Rotation.from_quat(body.reshape(-1, 4)).as_matrix().reshape(
        body.shape[:-1] + (3, 3)
    ) @ transform


def task_action_to_local(action: np.ndarray, start_site_rotation: np.ndarray) -> np.ndarray:
    """Map base-frame ``[H,7]`` actions to one fixed prediction-start EE frame."""
    values = np.asarray(action)
    rotation = np.asarray(start_site_rotation, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != ACTION_DIM or not np.isfinite(values).all():
        raise ValueError(f"action must be finite [H,{ACTION_DIM}]")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("start_site_rotation must be finite [3,3]")
    result = values.astype(np.float64, copy=True)
    result[:, :3] = result[:, :3] @ rotation
    result[:, 3:6] = result[:, 3:6] @ rotation
    result[:, :6] = bound_pose_action(result[:, :6])
    if np.any(np.abs(result[:, 6]) > 1.000001):
        raise ValueError("gripper action lies outside [-1,1]")
    return result.astype(np.float32)


def _natural_demo_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[1]), name
    except (IndexError, ValueError) as exc:
        raise ValueError(f"demo group must end in an integer: {name!r}") from exc


def _read_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Read sorted or repeated HDF5 rows without relying on h5py fancy-index rules."""
    indices = np.asarray(indices, dtype=np.int64)
    unique, inverse = np.unique(indices, return_inverse=True)
    return np.asarray(dataset[unique])[inverse]


def _validate_env_args(data: h5py.Group, path: Path) -> dict[str, object]:
    raw = data.attrs.get("env_args")
    if raw is None:
        raise ValueError(f"{path} is missing data.env_args")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    payload = json.loads(str(raw))
    kwargs = payload.get("env_kwargs", {})
    supported_environments = {
        ("Libero_Tabletop_Manipulation", "1.4.1"),
        ("Libero_Floor_Manipulation", "1.5.1"),
        ("Libero_Kitchen_Tabletop_Manipulation", "1.5.1"),
    }
    environment = (payload.get("env_name"), payload.get("env_version"))
    if environment not in supported_environments:
        raise ValueError(f"{path} has an incompatible LIBERO environment")
    if kwargs.get("robots") != ["Panda"]:
        raise ValueError(f"{path} was not recorded with one Panda robot")
    if kwargs.get("control_freq") != 20:
        raise ValueError(f"{path} was not recorded at 20 Hz")
    controller = kwargs.get("controller_configs")
    if not isinstance(controller, dict):
        raise ValueError(f"{path} is missing controller_configs")
    if controller.get("type") in {"BASIC", "SWITCHABLE"}:
        controller = controller.get("body_parts", {}).get("right")
        if not isinstance(controller, dict):
            raise ValueError(f"{path} is missing the right-arm controller config")
    for key, expected in RECORDED_CONTROLLER_CONFIG.items():
        actual = (
            controller.get("input_type") == "delta"
            if key == "control_delta" and "control_delta" not in controller
            else controller.get(key)
        )
        if isinstance(expected, list):
            matches = actual is not None and np.asarray(actual).shape == np.asarray(expected).shape
            matches = matches and bool(np.allclose(actual, expected))
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"{path} controller {key}={actual!r}, expected {expected!r}"
            )
    return {
        "env_name": payload["env_name"],
        "env_version": payload["env_version"],
        "robot": "Panda",
        "recorded_controller_config": dict(RECORDED_CONTROLLER_CONFIG),
        "runtime_requirements": dict(RUNTIME_CONTROLLER_REQUIREMENTS),
    }


@dataclass(frozen=True)
class TaskEpisode:
    path: Path
    task_name: str
    task_id: int
    demo_name: str
    length: int
    body_to_site_rotation: np.ndarray

    @property
    def key(self) -> str:
        return f"{self.task_name}:{self.demo_name}"


class LiberoSpatialImageDataset(BaseImageDataset):
    """Complete task-action windows, split only at episode boundaries."""

    def __init__(
        self,
        data_root: str | os.PathLike[str],
        *,
        split_path: str | os.PathLike[str],
        task_names: list[str] | tuple[str, ...],
        split: str = "train",
        horizon: int = 32,
        n_obs_steps: int = 2,
        proprio_representation: str,
    ) -> None:
        """Select one split's demos from the per-task LIBERO-Spatial HDF5 files.

        Args:
            data_root: directory holding ``<task_name>_demo_pc_instance_sg_50.hdf5`` per task.
            split_path: JSON ``{task_name: {"validation": [int], "test": [int]}}``; ids are the
                ``demo_<id>`` numbers held out of training. ``train`` is every other demo in the file.
            task_names: task vocabulary; ``task_id`` is the index in this list.
            split: ``train`` or ``validation``.
            horizon: action window length H; an episode yields ``T - H + 1`` windows.
            n_obs_steps: observation history length To.
            proprio_representation: must be ``ee_local_current_v1``.
        """
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if horizon < 1 or n_obs_steps < 1:
            raise ValueError("horizon and n_obs_steps must be positive")
        if proprio_representation != EE_LOCAL_PROPRIO:
            raise ValueError(
                f"proprio_representation must be {EE_LOCAL_PROPRIO!r}"
            )
        self.data_root = Path(data_root).resolve()
        self.split_path = Path(split_path).resolve()
        self.task_names = tuple(task_names)
        if len(set(self.task_names)) != len(self.task_names):
            raise ValueError("task vocabulary contains duplicates")
        self.split = split
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.proprio_representation = proprio_representation
        self.controller_contract: dict[str, object] | None = None
        self.environment_versions: set[tuple[str, str]] = set()
        self.episodes = self._select_episodes()
        self.episode_keys = tuple(episode.key for episode in self.episodes)
        self.indices = [
            (episode_index, start)
            for episode_index, episode in enumerate(self.episodes)
            for start in range(episode.length - self.horizon + 1)
        ]
        if not self.episodes or not self.indices:
            raise ValueError(f"{split} split has no complete action windows")
        self._handles: dict[Path, h5py.File] = {}
        self._handle_pid = os.getpid()
        self._proprio_stats: dict[str, np.ndarray] | None = None

    def _check_controller_contract(self, contract: dict[str, object]) -> None:
        self.environment_versions.add(
            (str(contract["env_name"]), str(contract["env_version"]))
        )
        controller_contract = {
            key: value
            for key, value in contract.items()
            if key not in {"env_name", "env_version"}
        }
        if self.controller_contract is None:
            self.controller_contract = controller_contract
        elif self.controller_contract != controller_contract:
            raise ValueError("LIBERO task files use different controller contracts")

    def _select_episodes(self) -> list[TaskEpisode]:
        held_out = json.loads(self.split_path.read_text(encoding="utf-8"))
        if set(held_out) != set(self.task_names):
            raise ValueError("split file tasks differ from the configured vocabulary")
        episodes: list[TaskEpisode] = []
        for task_id, task_name in enumerate(self.task_names):
            path = self.data_root / f"{task_name}{DATA_FILE_SUFFIX}"
            if not path.is_file():
                raise FileNotFoundError(path)
            held = held_out[task_name]
            task_episodes, contract = self._scan_file(
                path, task_name, task_id, held["validation"], held["test"]
            )
            self._check_controller_contract(contract)
            episodes.extend(task_episodes)
        return episodes

    def _scan_file(
        self,
        path: Path,
        task_name: str,
        task_id: int,
        validation_ids: list[int],
        test_ids: list[int],
    ) -> tuple[list[TaskEpisode], dict[str, object]]:
        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                raise ValueError(f"{path} has no data group")
            data = handle["data"]
            controller_contract = _validate_env_args(data, path)
            validation = {f"demo_{i}" for i in validation_ids}
            test = {f"demo_{i}" for i in test_ids}
            missing = sorted((validation | test) - set(data.keys()))
            if missing or validation & test:
                raise ValueError(f"{path}: held-out {missing} missing or validation/test overlap")
            demo_names = sorted(
                validation if self.split == "validation" else set(data.keys()) - validation - test,
                key=_natural_demo_key,
            )
            transforms = []
            for name in {demo_names[0], demo_names[-1]}:
                model_xml = data[name].attrs.get("model_file")
                if isinstance(model_xml, bytes):
                    model_xml = model_xml.decode("utf-8")
                if not isinstance(model_xml, str):
                    raise ValueError(f"{path}:{name} is missing model_file XML")
                transforms.append(body_to_site_rotation_from_xml(model_xml))
            if any(
                not np.allclose(transforms[0], value, atol=1e-7, rtol=1e-7)
                for value in transforms[1:]
            ):
                raise ValueError(f"{path} body-to-site transform varies across demos")
            transform = transforms[0]
            episodes = []
            for demo_name in demo_names:
                demo = data[demo_name]
                length = int(demo["actions"].shape[0])
                if demo["actions"].shape != (length, ACTION_DIM):
                    raise ValueError(
                        f"{path}:{demo_name} actions must be [T,{ACTION_DIM}]"
                    )
                if length < self.horizon:
                    raise ValueError(f"{path}:{demo_name} is shorter than horizon")
                obs = demo.get("obs")
                if obs is None:
                    raise ValueError(f"{path}:{demo_name} is missing obs")
                for key in IMAGE_KEYS + PROPRIO_KEYS:
                    if key not in obs or obs[key].shape[0] != length:
                        raise ValueError(f"{path}:{demo_name} has invalid observation {key}")
                episodes.append(TaskEpisode(
                    path=path,
                    task_name=task_name,
                    task_id=task_id,
                    demo_name=demo_name,
                    length=length,
                    body_to_site_rotation=transform.copy(),
                ))
            return episodes, controller_contract

    def _handle(self, path: Path) -> h5py.File:
        pid = os.getpid()
        if pid != self._handle_pid:
            self._handles = {}
            self._handle_pid = pid
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def _absolute_proprio(
        obs: h5py.Group, indices: np.ndarray, transform: np.ndarray
    ) -> np.ndarray:
        position = np.asarray(_read_rows(obs["robot0_eef_pos"], indices), dtype=np.float32)
        body_quaternion = np.asarray(_read_rows(obs["robot0_eef_quat"], indices), dtype=np.float64)
        site_rotation = site_rotation_from_body_quaternion(body_quaternion, transform)
        gripper = np.asarray(_read_rows(obs["robot0_gripper_qpos"], indices), dtype=np.float32)
        return np.concatenate(
            (position, site_rotation.reshape(len(indices), 9).astype(np.float32), gripper), axis=-1
        )

    @classmethod
    def _proprio(
        cls, obs: h5py.Group, indices: np.ndarray, transform: np.ndarray
    ) -> np.ndarray:
        return ee_local_proprio(cls._absolute_proprio(obs, indices, transform))

    def __getitem__(self, index: int) -> dict[str, object]:
        episode_index, start = self.indices[index]
        episode = self.episodes[episode_index]
        demo = self._handle(episode.path)["data"][episode.demo_name]
        obs = demo["obs"]
        history = np.clip(
            np.arange(start - self.n_obs_steps + 1, start + 1), 0, episode.length - 1
        )
        images = {}
        for source, target in zip(IMAGE_KEYS, ("agentview_rgb", "wrist_rgb")):
            value = np.asarray(_read_rows(obs[source], history), dtype=np.uint8)
            images[target] = torch.from_numpy(np.ascontiguousarray(value)).permute(0, 3, 1, 2).float().div_(255.)
        absolute_proprio = self._absolute_proprio(
            obs, history, episode.body_to_site_rotation
        )
        proprio = ee_local_proprio(absolute_proprio)
        action = np.asarray(demo["actions"][start:start + self.horizon], dtype=np.float32)
        start_rotation = absolute_proprio[-1, 3:12].reshape(3, 3)
        local_action = task_action_to_local(action, start_rotation)
        return {
            "obs": {
                **images,
                "proprio": torch.from_numpy(proprio),
                "task_id": torch.tensor(episode.task_id, dtype=torch.long),
            },
            "action": torch.from_numpy(local_action),
        }

    def proprio_stats(self) -> dict[str, np.ndarray]:
        if self._proprio_stats is None:
            values = []
            for episode in self.episodes:
                obs = self._handle(episode.path)["data"][episode.demo_name]["obs"]
                absolute = self._absolute_proprio(
                    obs,
                    np.arange(episode.length, dtype=np.int64),
                    episode.body_to_site_rotation,
                )
                for start in range(episode.length - self.horizon + 1):
                    history = np.clip(
                        np.arange(start - self.n_obs_steps + 1, start + 1),
                        0,
                        episode.length - 1,
                    )
                    values.append(ee_local_proprio(absolute[history]))
            stacked = np.concatenate(values, axis=0).astype(np.float64)
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            std[std < 1e-6] = 1.0
            self._proprio_stats = {
                "min": stacked.min(axis=0).astype(np.float32),
                "max": stacked.max(axis=0).astype(np.float32),
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
            }
        return {key: value.copy() for key, value in self._proprio_stats.items()}

    def get_validation_dataset(self) -> "LiberoSpatialImageDataset":
        return type(self)(
            data_root=self.data_root,
            split_path=self.split_path,
            task_names=self.task_names,
            split="validation",
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            proprio_representation=self.proprio_representation,
        )

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        stats = self.proprio_stats()
        dtype = torch.float32
        scale = np.ones(PROPRIO_DIM, dtype=np.float32)
        offset = np.zeros(PROPRIO_DIM, dtype=np.float32)
        standardized = np.r_[0:3, 12:14]
        scale[standardized] = 1.0 / stats["std"][standardized]
        offset[standardized] = -stats["mean"][standardized] / stats["std"][standardized]
        proprio = SingleFieldLinearNormalizer.create_manual(
            scale=torch.from_numpy(scale).to(dtype),
            offset=torch.from_numpy(offset).to(dtype),
            input_stats_dict={
                key: torch.from_numpy(value).to(dtype) for key, value in stats.items()
            },
        )
        normalizer = LinearNormalizer()
        normalizer["agentview_rgb"] = SingleFieldLinearNormalizer.create_identity(dtype)
        normalizer["wrist_rgb"] = SingleFieldLinearNormalizer.create_identity(dtype)
        normalizer["proprio"] = proprio
        normalizer["action"] = SingleFieldLinearNormalizer.create_identity(dtype)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError(
            "local-frame actions depend on each window start; use dataset windows"
        )

    def identity(self) -> dict[str, object]:
        counts = {}
        body_to_site_rotation_by_task = {}
        for episode in self.episodes:
            item = counts.setdefault(episode.task_name, {"episodes": 0, "steps": 0, "windows": 0})
            item["episodes"] += 1
            item["steps"] += episode.length
            item["windows"] += episode.length - self.horizon + 1
            rotation = episode.body_to_site_rotation.tolist()
            previous = body_to_site_rotation_by_task.setdefault(episode.task_name, rotation)
            if not np.allclose(previous, rotation, atol=1e-7, rtol=1e-7):
                raise ValueError("body-to-site transform varies within one task split")
        return {
            "data_root": str(self.data_root),
            "split_path": str(self.split_path),
            "split": self.split,
            "task_names": list(self.task_names),
            "task_id_rule": "index in task_names",
            "proprio_representation": self.proprio_representation,
            "horizon": self.horizon,
            "n_obs_steps": self.n_obs_steps,
            "body_to_site_rotation_by_task": body_to_site_rotation_by_task,
            "action_representation": {
                "frame": "prediction_start_eef",
                "channels": "delta_xyz_delta_axis_angle_gripper",
                "pose_bound": "translation_and_rotation_independent_l2_unit_balls",
            },
            "controller_contract": self.controller_contract,
            "environment_versions": [
                {"env_name": name, "env_version": version}
                for name, version in sorted(self.environment_versions)
            ],
            "observation_keys": {
                "images": list(IMAGE_KEYS),
                "proprio": list(PROPRIO_KEYS),
            },
            "episodes": len(self.episodes),
            "windows": len(self),
            "counts_by_task": counts,
            "episode_keys": list(self.episode_keys),
        }
