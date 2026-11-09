"""Continuous noise-to-action Flow Matching shared by both experts."""

from __future__ import annotations

from collections.abc import Callable

import torch


PI05_TIME_CONVENTION = "pi05_t1_noise_t0_action"


def sample_generation_time(
    reference: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``Beta(1.5, 1) * .999 + .001`` once per batch item."""
    if reference.ndim < 1:
        raise ValueError("reference must have a batch dimension")
    uniform = torch.rand(
        reference.shape[0],
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )
    return uniform.pow(2.0 / 3.0).mul(0.999).add(0.001)


def training_pair(
    action: torch.Tensor,
    noise: torch.Tensor,
    generation_time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct ``x_t=t*noise+(1-t)*action`` and ``u=noise-action``."""
    if action.ndim != 3 or noise.shape != action.shape:
        raise ValueError("action and noise must have the same [B,H,D] shape")
    if generation_time.shape != (action.shape[0],):
        raise ValueError("generation_time must be [B]")
    if not torch.isfinite(generation_time).all() or torch.any(
        (generation_time < 0) | (generation_time > 1)
    ):
        raise ValueError("generation_time must be finite in [0,1]")
    time = generation_time[:, None, None]
    return time * noise + (1.0 - time) * action, noise - action


def euler_sample(
    velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    noise: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """Integrate one shared state from noise at ``t=1`` to action at ``t=0``."""
    if noise.ndim != 3:
        raise ValueError("noise must be [B,H,D]")
    if type(num_steps) is not int or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    state = noise.clone()
    delta = -1.0 / num_steps
    for step in range(num_steps):
        time = state.new_full((state.shape[0],), 1.0 + step * delta)
        field = velocity(state, time)
        if field.shape != state.shape:
            raise ValueError("velocity must preserve [B,H,D]")
        state = state + delta * field
    return state
