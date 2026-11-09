"""DP-FM image task expert with a public continuous velocity field."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.pose_util import EE_LOCAL_PROPRIO
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.flow.continuous_flow import (
    PI05_TIME_CONVENTION,
    euler_sample,
    sample_generation_time,
    training_pair,
)
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class FlowMatchUnetImagePolicy(BaseImagePolicy):
    """Image-conditioned seven-channel task field.

    ``horizon`` is the generated trajectory length. ``n_action_steps`` is the
    prefix returned to a runner after generating the complete trajectory.

    ``agentview_dropout`` is the per-sample probability, in ``[0,1)``, that
    training replaces every agentview frame of a sample with black (zeros).
    Setting ``mask_agentview`` to ``True`` blacks out agentview for every
    condition this policy encodes, which is how evaluation runs the policy on
    the wrist camera alone.
    """

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_tasks: int,
        proprio_representation: str,
        task_embedding_dim: int = 32,
        num_inference_steps: int = 10,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple[int, ...] = (128, 256),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        flow_time_convention: str = PI05_TIME_CONVENTION,
        agentview_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if flow_time_convention != PI05_TIME_CONVENTION:
            raise ValueError(f"unsupported flow convention {flow_time_convention!r}")
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon must be positive")
        if type(n_action_steps) is not int or not 1 <= n_action_steps <= horizon:
            raise ValueError("n_action_steps must lie in [1,horizon]")
        if type(n_obs_steps) is not int or n_obs_steps < 1:
            raise ValueError("n_obs_steps must be positive")
        if type(num_tasks) is not int or num_tasks < 1:
            raise ValueError("num_tasks must be positive")
        if proprio_representation != EE_LOCAL_PROPRIO:
            raise ValueError(
                f"proprio_representation must be {EE_LOCAL_PROPRIO!r}"
            )
        if not 0.0 <= agentview_dropout < 1.0:
            raise ValueError("agentview_dropout must lie in [0,1)")
        if horizon % (2 ** (len(down_dims) - 1)):
            raise ValueError("horizon must survive U-Net downsampling")
        action_shape = tuple(shape_meta["action"]["shape"])
        if action_shape != (7,):
            raise ValueError("task action must be pose6 plus gripper [7]")
        obs_meta = shape_meta["obs"]
        expected_obs = {"agentview_rgb", "wrist_rgb", "proprio", "task_id"}
        if set(obs_meta) != expected_obs:
            raise ValueError(f"task observations must be {sorted(expected_obs)}")
        if tuple(obs_meta["task_id"]["shape"]) != ():
            raise ValueError("task_id must be scalar")
        if obs_meta["task_id"].get("type") != "categorical":
            raise ValueError("task_id must use categorical encoding")
        obs_feature_dim = int(obs_encoder.output_shape()[0])
        condition_dim = obs_feature_dim * n_obs_steps + task_embedding_dim
        self.model = ConditionalUnet1D(
            input_dim=action_shape[0],
            global_cond_dim=condition_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.obs_encoder = obs_encoder
        self.task_embedding = nn.Embedding(num_tasks, task_embedding_dim)
        self.normalizer = LinearNormalizer()
        self.shape_meta = deepcopy(shape_meta)
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_tasks = num_tasks
        self.proprio_representation = proprio_representation
        self.action_dim = action_shape[0]
        self.obs_feature_dim = obs_feature_dim
        self.condition_dim = condition_dim
        self.num_inference_steps = num_inference_steps
        self.flow_time_convention = flow_time_convention
        self.agentview_dropout = agentview_dropout
        self.mask_agentview = False

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def encode_condition(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        expected = set(self.shape_meta["obs"])
        if set(obs_dict) != expected:
            raise ValueError(f"task observation keys must be {sorted(expected)}")
        task_id = obs_dict["task_id"]
        if task_id.ndim != 1 or task_id.dtype != torch.long:
            raise ValueError("task_id must be int64 [B]")
        if torch.any((task_id < 0) | (task_id >= self.num_tasks)):
            raise ValueError("task_id is outside the configured vocabulary")
        continuous_obs = {key: value for key, value in obs_dict.items() if key != "task_id"}
        if self.mask_agentview:
            continuous_obs["agentview_rgb"] = torch.zeros_like(
                continuous_obs["agentview_rgb"]
            )
        normalized = self.normalizer.normalize(continuous_obs)
        first = next(iter(normalized.values()))
        batch = first.shape[0]
        for key, value in normalized.items():
            if value.shape[0] != batch or value.shape[1] < self.n_obs_steps:
                raise ValueError(f"{key} does not contain the configured observation history")
        flattened = dict_apply(
            normalized,
            lambda value: value[:, : self.n_obs_steps].reshape(-1, *value.shape[2:]),
        )
        obs_features = self.obs_encoder(flattened).reshape(batch, -1)
        condition = torch.cat((obs_features, self.task_embedding(task_id)), dim=-1)
        if condition.shape != (batch, self.condition_dim):
            raise RuntimeError("encoded task condition has an unexpected shape")
        return condition

    def velocity(
        self,
        state: torch.Tensor,
        generation_time: torch.Tensor,
        *,
        obs: Dict[str, torch.Tensor] | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state.ndim != 3 or state.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"state must be [B,{self.horizon},{self.action_dim}]")
        if generation_time.shape != (state.shape[0],):
            raise ValueError("generation_time must be [B]")
        if (obs is None) == (condition is None):
            raise ValueError("pass exactly one of obs or condition")
        condition = self.encode_condition(obs) if condition is None else condition
        if condition.shape != (state.shape[0], self.condition_dim):
            raise ValueError("condition shape does not match the state batch")
        return self.model(state, generation_time, global_cond=condition)

    def sample(
        self,
        obs_dict: Dict[str, torch.Tensor],
        *,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        condition = self.encode_condition(obs_dict)
        shape = (condition.shape[0], self.horizon, self.action_dim)
        if noise is None:
            noise = torch.randn(
                shape,
                device=condition.device,
                dtype=condition.dtype,
                generator=generator,
            )
        if noise.shape != shape:
            raise ValueError(f"noise must be {shape}")
        steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        return euler_sample(
            lambda state, time: self.velocity(state, time, condition=condition),
            noise,
            steps,
        )

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalized_action = self.sample(obs_dict)
        action_pred = self.normalizer["action"].unnormalize(normalized_action)
        return {
            "action": action_pred[:, : self.n_action_steps],
            "action_pred": action_pred,
        }

    def compute_loss(
        self,
        batch: dict[str, object],
        *,
        noise: torch.Tensor | None = None,
        generation_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if set(batch) != {"obs", "action"}:
            raise ValueError("batch must contain exactly obs and action")
        action = batch["action"]
        if not isinstance(action, torch.Tensor) or action.shape[1:] != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError(f"action must be [B,{self.horizon},{self.action_dim}]")
        normalized_action = self.normalizer["action"].normalize(action)
        noise = torch.randn_like(normalized_action) if noise is None else noise
        if noise.shape != normalized_action.shape:
            raise ValueError("noise must match action")
        generation_time = (
            sample_generation_time(normalized_action)
            if generation_time is None
            else generation_time
        )
        obs = batch["obs"]
        if self.training and self.agentview_dropout > 0.0:
            agentview = obs["agentview_rgb"]
            keep = torch.rand(agentview.shape[0], device=agentview.device) >= self.agentview_dropout
            obs = {
                **obs,
                "agentview_rgb": agentview * keep.view(-1, *([1] * (agentview.ndim - 1))),
            }
        state, target = training_pair(normalized_action, noise, generation_time)
        prediction = self.velocity(
            state,
            generation_time,
            condition=self.encode_condition(obs),
        )
        return F.mse_loss(prediction, target)
