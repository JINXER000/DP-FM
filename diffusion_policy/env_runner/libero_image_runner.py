"""Batched-policy, asynchronous multi-process LIBERO image runner."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
from typing import Mapping, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from diffusion_policy.common.pose_util import (
    EE_LOCAL_PROPRIO,
    bound_pose_action,
    ee_action_to_base,
    ee_local_proprio,
)
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")


@dataclass(frozen=True)
class LiberoEpisode:
    task_name: str
    init_state_index: int
    sampling_seed: int
    scenario: str = "default"
    bddl_file: str | None = None
    init_states_path: str | None = None


def _part_controller(env):
    return env.robots[0].composite_controller.part_controllers["right"]


def _verify_controller(env) -> None:
    controller = _part_controller(env)
    expected_scale = np.asarray([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
    if not (
        controller.input_type == "delta"
        and controller.input_ref_frame == "base"
        and controller._goal_update_mode == "achieved"
        and env.control_freq == 20
    ):
        raise ValueError("runner requires achieved-state base-frame delta OSC at 20 Hz")
    if not np.allclose(controller.output_max, expected_scale):
        raise ValueError(
            "LIBERO controller pose scale differs from the training contract"
        )
    if not np.allclose(controller.origin_ori, np.eye(3), atol=1e-8):
        raise ValueError("task action frame assumes controller base equals world axes")


def make_libero_env(
    task_name: str,
    camera_size: int | None = 128,
    *,
    bddl_file: str | None = None,
    init_states_path: str | None = None,
):
    """Create an Object, Spatial, or explicitly configured custom environment."""
    from libero.libero import benchmark
    from libero.libero.envs.env_wrapper import ControlEnv

    if (bddl_file is None) != (init_states_path is None):
        raise ValueError("bddl_file and init_states_path must be supplied together")
    if bddl_file is None:
        for suite_name in ("libero_spatial", "libero_object"):
            suite = benchmark.get_benchmark_dict()[suite_name]()
            names = [suite.get_task(index).name for index in range(suite.n_tasks)]
            if task_name in names:
                break
        else:
            raise ValueError(
                f"task {task_name!r} is absent from libero_spatial and libero_object"
            )
        task_index = names.index(task_name)
        resolved_bddl = suite.get_task_bddl_file_path(task_index)
        initial_states = np.asarray(suite.get_task_init_states(task_index))
    else:
        resolved_bddl = str(Path(bddl_file).expanduser().resolve())
        # LIBERO saves (N, 92) tensors; LIBERO-plus layout variants save one (92,) numpy state.
        initial_states = np.atleast_2d(
            np.asarray(
                torch.load(
                    Path(init_states_path).expanduser().resolve(),
                    map_location="cpu",
                    weights_only=False,
                )
            )
        )
        if initial_states.ndim != 2 or not np.isfinite(initial_states).all():
            raise ValueError("custom initial states must be a finite rank-1 or rank-2 tensor")
    options = dict(
        bddl_file_name=resolved_bddl,
        use_camera_obs=camera_size is not None,
        has_renderer=False,
        has_offscreen_renderer=camera_size is not None,
        control_freq=20,
        horizon=10000,
        ignore_done=True,
    )
    if camera_size is not None:
        options.update(
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=camera_size,
            camera_widths=camera_size,
        )
    wrapper = ControlEnv(**options)
    try:
        wrapper.reset()
        _verify_controller(wrapper.env)
    except Exception:
        wrapper.close()
        raise
    return wrapper, initial_states


def _reset(wrapper, initial_state: np.ndarray) -> dict[str, np.ndarray]:
    """Restore ``initial_state``; the warm-up matches openpi ``examples/libero/main.py`` and is outside ``max_steps``."""
    wrapper.reset()
    wrapper.env.sim.set_state_from_flattened(np.asarray(initial_state).copy())
    wrapper.env.sim.forward()
    observation = None
    for _ in range(10):
        observation, _, _, _ = wrapper.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
    return observation


def restore_state(env, state: np.ndarray) -> None:
    """Set the MuJoCo state to a finite flattened ``state`` and run forward kinematics."""
    expected = env.sim.get_state().flatten().shape
    if state.shape != expected or not np.isfinite(state).all():
        raise ValueError(f"simulator state must be finite {expected}")
    env.sim.set_state_from_flattened(state.copy())
    env.sim.forward()


def step_from_state(wrapper, state: np.ndarray, action: np.ndarray):
    """Restore ``state`` and execute one base-frame ``[7]`` action from it.

    The OSC goal and gripper command are reset to the restored pose, so the
    step reproduces a recorded ``states[t] -> actions[t]`` transition.
    Returns the wrapper's ``(observation, reward, done, info)``.
    """
    wrapper.reset()
    env = wrapper.env
    restore_state(env, state)
    controller = _part_controller(env)
    controller.update(force=True)
    joints = np.asarray(env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes]).copy()
    controller.update_initial_joints(joints)
    controller.reset_goal()
    env.robots[0].gripper["right"].current_action[:] = 0
    env._update_observables(force=True)
    return wrapper.step(np.asarray(action, dtype=np.float64))


def task_frame(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {}
    for source, target in zip(CAMERA_KEYS, ("agentview_rgb", "wrist_rgb")):
        image = np.asarray(observation[source])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{source} must be [H,W,3]")
        result[target] = (
            np.ascontiguousarray(image[::-1].transpose(2, 0, 1)).astype(np.float32)
            / 255.0
        )
    position = np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
    quaternion = np.asarray(observation["robot0_eef_quat_site"], dtype=np.float64)
    rotation = Rotation.from_quat(quaternion).as_matrix().astype(np.float32)
    gripper = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32)
    if position.shape != (3,) or quaternion.shape != (4,) or gripper.shape != (2,):
        raise ValueError("runtime proprio shapes differ from training")
    result["proprio"] = np.concatenate((position, rotation.reshape(9), gripper))
    result["eef_pose"] = np.concatenate((position, quaternion.astype(np.float32)))
    return result


def _video_frame(frame: dict[str, np.ndarray]) -> np.ndarray:
    views = []
    for key in ("agentview_rgb", "wrist_rgb"):
        image = np.asarray(frame[key])
        views.append(
            np.clip(image.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        )
    return np.concatenate(views, axis=1)


def _batch_observation(
    slots: list[dict[str, object]],
    device: torch.device,
) -> dict:
    result = {}
    for key in ("agentview_rgb", "wrist_rgb"):
        result[key] = torch.from_numpy(
            np.stack(
                [np.stack([frame[key] for frame in slot["history"]]) for slot in slots]
            )
        ).to(device)
    result["proprio"] = torch.from_numpy(
        np.stack(
            [
                ee_local_proprio(
                    np.stack([frame["proprio"] for frame in slot["history"]])
                )
                for slot in slots
            ]
        )
    ).to(device)
    result["task_id"] = torch.tensor(
        [slot["task_id"] for slot in slots], dtype=torch.long, device=device
    )
    return result


def _environment_worker(
    connection,
    episode: LiberoEpisode,
    camera_size: int,
    max_steps: int,
    render_backend: str,
) -> None:
    wrapper = None
    try:
        os.environ["MUJOCO_GL"] = render_backend
        wrapper, initial_states = make_libero_env(
            episode.task_name,
            camera_size,
            bddl_file=episode.bddl_file,
            init_states_path=episode.init_states_path,
        )
        if not 0 <= episode.init_state_index < len(initial_states):
            raise ValueError("init_state_index is outside the LIBERO task states")
        observation = _reset(wrapper, initial_states[episode.init_state_index])
        frame = task_frame(observation)
        connection.send({"kind": "ready", "frame": frame})
        steps = 0
        success_step = None
        while True:
            command = connection.recv()
            if command["kind"] == "close":
                break
            if command["kind"] != "step":
                raise ValueError(f"unknown worker command {command['kind']!r}")
            action_chunk = np.asarray(command["actions"], dtype=np.float32)
            if action_chunk.ndim != 2 or action_chunk.shape[1] != 7:
                raise ValueError("worker actions must be [K,7]")
            start_pose = frame["eef_pose"]
            frames = []
            simulator_seconds = []
            env_actions = []
            for local in action_chunk[: max_steps - steps]:
                pose = ee_action_to_base(bound_pose_action(local[:6]), start_pose)
                env_action = np.r_[pose, np.clip(local[6], -1.0, 1.0)]
                started = time.perf_counter()
                observation, _, _, _ = wrapper.step(np.clip(env_action, -1.0, 1.0))
                simulator_seconds.append(time.perf_counter() - started)
                env_actions.append(env_action.astype(np.float32))
                frame = task_frame(observation)
                frames.append(frame)
                steps += 1
                if wrapper.check_success():
                    success_step = steps
                    break
            connection.send(
                {
                    "kind": "step_result",
                    "frames": frames,
                    "simulator_seconds": simulator_seconds,
                    "env_actions": env_actions,
                    "steps": steps,
                    "success_step": success_step,
                    "done": success_step is not None or steps >= max_steps,
                }
            )
    except BaseException:
        try:
            connection.send({"kind": "error", "traceback": traceback.format_exc()})
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if wrapper is not None:
            wrapper.close()
        connection.close()


class LiberoImageRunner(BaseImageRunner):
    """Batch policy inference while stepping independent simulators concurrently."""

    def __init__(
        self,
        output_dir: str,
        task_vocabulary: Sequence[str],
        *,
        episodes: Sequence[LiberoEpisode] | None = None,
        init_state_indices: Sequence[int] | None = None,
        sampling_seed: int = 0,
        bddl_file: str | None = None,
        init_states_path: str | None = None,
        scenario_initial_states: Mapping[str, str] | None = None,
        episodes_per_scenario: int = 3,
        num_envs: int = 1,
        prediction_horizon: int = 32,
        n_action_steps: int = 10,
        n_obs_steps: int = 2,
        proprio_representation: str = EE_LOCAL_PROPRIO,
        max_steps: int = 300,
        camera_size: int = 128,
        execution_action_steps: int | None = None,
        record_video: bool = False,
        render_backend: str = "osmesa",
    ) -> None:
        super().__init__(output_dir)
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be positive")
        if not 1 <= n_action_steps <= prediction_horizon:
            raise ValueError("n_action_steps must lie in [1,prediction_horizon]")
        if execution_action_steps is None:
            execution_action_steps = n_action_steps
        if not 1 <= execution_action_steps <= n_action_steps:
            raise ValueError("execution_action_steps must lie in [1,n_action_steps]")
        if proprio_representation != EE_LOCAL_PROPRIO:
            raise ValueError(
                f"proprio_representation must be {EE_LOCAL_PROPRIO!r}"
            )
        if render_backend not in {"osmesa", "egl"}:
            raise ValueError("render_backend must be 'osmesa' or 'egl'")
        vocabulary = tuple(task_vocabulary)
        if not vocabulary:
            raise ValueError("task_vocabulary must not be empty")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("task_vocabulary contains duplicates")
        if scenario_initial_states is not None:
            if episodes is not None or init_state_indices is not None:
                raise ValueError(
                    "scenario_initial_states cannot be combined with explicit episodes"
                )
            if init_states_path is not None or bddl_file is None:
                raise ValueError(
                    "scenario evaluation requires bddl_file and per-scenario states"
                )
            if len(vocabulary) != 1:
                raise ValueError("scenario evaluation requires one task vocabulary entry")
            if episodes_per_scenario < 1:
                raise ValueError("episodes_per_scenario must be positive")
            scenarios = tuple(scenario_initial_states.items())
            if not scenarios:
                raise ValueError("scenario_initial_states must not be empty")
            episodes = [
                LiberoEpisode(
                    task_name=vocabulary[0],
                    init_state_index=episode_index,
                    sampling_seed=(
                        sampling_seed
                        + scenario_index * episodes_per_scenario
                        + episode_index
                    ),
                    scenario=str(scenario_name),
                    bddl_file=bddl_file,
                    init_states_path=str(states_path),
                )
                for scenario_index, (scenario_name, states_path) in enumerate(scenarios)
                for episode_index in range(episodes_per_scenario)
            ]
        elif episodes is None:
            if (bddl_file is None) != (init_states_path is None):
                raise ValueError(
                    "bddl_file and init_states_path must be supplied together"
                )
            indices = tuple(init_state_indices or (0,))
            if not indices or any(
                type(index) is not int or index < 0 for index in indices
            ):
                raise ValueError(
                    "init_state_indices must contain non-negative integers"
                )
            episodes = [
                LiberoEpisode(
                    task_name=task_name,
                    init_state_index=state_index,
                    sampling_seed=sampling_seed + task_index * len(indices) + offset,
                    bddl_file=bddl_file,
                    init_states_path=init_states_path,
                )
                for task_index, task_name in enumerate(vocabulary)
                for offset, state_index in enumerate(indices)
            ]
        elif init_state_indices is not None:
            raise ValueError(
                "init_state_indices cannot be combined with explicit episodes"
            )
        elif bddl_file is not None:
            raise ValueError("custom environment paths belong on explicit episodes")
        if not episodes:
            raise ValueError("episodes must not be empty")
        unknown = sorted({episode.task_name for episode in episodes} - set(vocabulary))
        if unknown:
            raise ValueError(f"episode tasks absent from policy vocabulary: {unknown}")
        self.episodes = list(episodes)
        self.task_vocabulary = vocabulary
        self.num_envs = num_envs
        self.prediction_horizon = prediction_horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.proprio_representation = proprio_representation
        self.max_steps = max_steps
        self.camera_size = camera_size
        self.execution_action_steps = execution_action_steps
        self.record_video = bool(record_video)
        self.render_backend = render_backend

    def run(self, policy) -> dict[str, object]:
        if policy.horizon != self.prediction_horizon:
            raise ValueError("runner and policy prediction horizons differ")
        if policy.n_action_steps != self.n_action_steps:
            raise ValueError("runner and policy action horizons differ")
        if policy.n_obs_steps != self.n_obs_steps:
            raise ValueError("runner and policy observation histories differ")
        if policy.proprio_representation != self.proprio_representation:
            raise ValueError("runner and policy proprio representations differ")
        device = next(policy.parameters()).device
        dtype = next(policy.parameters()).dtype
        records = []
        for start in range(0, len(self.episodes), self.num_envs):
            chunk = self.episodes[start : start + self.num_envs]
            slots = self._open_slots(chunk, device)
            try:
                while any(not slot["done"] for slot in slots):
                    active = [slot for slot in slots if not slot["done"]]
                    obs = _batch_observation(active, device)
                    noise = torch.stack(
                        [
                            torch.randn(
                                (self.prediction_horizon, 7),
                                generator=slot["generator"],
                                device=device,
                                dtype=dtype,
                            )
                            for slot in active
                        ]
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        normalized = policy.sample(obs, noise=noise)
                        actions = policy.normalizer["action"].unnormalize(normalized)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    latency = time.perf_counter() - started
                    for slot, action in zip(active, actions):
                        slot["inference_seconds"].append(latency)
                    self._step_slots(
                        active, actions[:, : self.execution_action_steps]
                    )
                records.extend(self._records(slots))
            finally:
                self._close_slots(slots)
        successes = sum(record["success"] for record in records)
        scenario_results = {}
        for scenario in dict.fromkeys(record["scenario"] for record in records):
            selected = [record for record in records if record["scenario"] == scenario]
            scenario_successes = sum(record["success"] for record in selected)
            scenario_results[scenario] = {
                "episodes": len(selected),
                "successes": scenario_successes,
                "success_rate": scenario_successes / len(selected),
            }
        return {
            "episodes": records,
            "successes": successes,
            "success_rate": successes / len(records),
            "scenarios": scenario_results,
        }

    def _open_slots(self, episodes, device: torch.device) -> list[dict[str, object]]:
        context = mp.get_context("spawn")
        slots = []
        try:
            for episode in episodes:
                parent_connection, child_connection = context.Pipe()
                process = context.Process(
                    target=_environment_worker,
                    args=(
                        child_connection,
                        episode,
                        self.camera_size,
                        self.max_steps,
                        self.render_backend,
                    ),
                    daemon=True,
                )
                process.start()
                child_connection.close()
                slots.append(
                    {
                        "episode": episode,
                        "connection": parent_connection,
                        "process": process,
                    }
                )
                slot = slots[-1]
                response = parent_connection.recv()
                self._raise_worker_error(response)
                if response["kind"] != "ready":
                    raise RuntimeError("environment worker did not send ready")
                frame = response["frame"]
                episode = slot["episode"]
                generator = torch.Generator(device=device)
                generator.manual_seed(episode.sampling_seed)
                slot.update(
                    history=deque(
                        [frame] * self.n_obs_steps, maxlen=self.n_obs_steps
                    ),
                    task_id=self.task_vocabulary.index(episode.task_name),
                    generator=generator,
                    done=False,
                    success_step=None,
                    steps=0,
                    inference_seconds=[],
                    simulator_seconds=[],
                    env_actions=[],
                    video_frames=(
                        [_video_frame(frame)] if self.record_video else None
                    ),
                )
        except BaseException:
            self._close_slots(slots)
            raise
        return slots

    @staticmethod
    def _raise_worker_error(response: dict[str, object]) -> None:
        if response.get("kind") == "error":
            raise RuntimeError("LIBERO environment worker failed:\n" + response["traceback"])

    def _step_slots(
        self,
        slots: list[dict[str, object]],
        action_chunks: torch.Tensor,
    ) -> None:
        chunks = action_chunks.detach().float().cpu().numpy()
        for slot, actions in zip(slots, chunks):
            slot["connection"].send({"kind": "step", "actions": actions})
        for slot in slots:
            response = slot["connection"].recv()
            self._raise_worker_error(response)
            if response["kind"] != "step_result":
                raise RuntimeError("environment worker returned an invalid result")
            frames = response["frames"]
            for frame in frames:
                slot["history"].append(frame)
                if slot["video_frames"] is not None:
                    slot["video_frames"].append(_video_frame(frame))
            slot["steps"] = response["steps"]
            slot["success_step"] = response["success_step"]
            slot["done"] = response["done"]
            slot["simulator_seconds"].extend(response["simulator_seconds"])
            slot["env_actions"].extend(
                action.tolist() for action in response["env_actions"]
            )

    @staticmethod
    def _close_slots(slots: list[dict[str, object]]) -> None:
        for slot in slots:
            connection = slot.get("connection")
            process = slot.get("process")
            if connection is not None and process is not None and process.is_alive():
                try:
                    connection.send({"kind": "close"})
                except (BrokenPipeError, EOFError):
                    pass
        for slot in slots:
            connection = slot.get("connection")
            process = slot.get("process")
            if connection is not None:
                connection.close()
            if process is not None:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)

    @staticmethod
    def _records(slots: list[dict[str, object]]) -> list[dict[str, object]]:
        records = []
        for slot in slots:
            episode = slot["episode"]
            record = {
                "task_name": episode.task_name,
                "scenario": episode.scenario,
                "init_state_index": episode.init_state_index,
                "sampling_seed": episode.sampling_seed,
                "success": slot["success_step"] is not None,
                "success_step": slot["success_step"],
                "steps": slot["steps"],
                "inference_seconds": list(slot["inference_seconds"]),
                "simulator_seconds": list(slot["simulator_seconds"]),
                "env_actions": list(slot["env_actions"]),
            }
            if slot["video_frames"] is not None:
                record["video_frames"] = np.stack(slot["video_frames"])
            records.append(record)
        return records


__all__ = [
    "LiberoEpisode",
    "LiberoImageRunner",
    "make_libero_env",
    "task_frame",
]
