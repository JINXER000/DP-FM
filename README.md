# DP-FM

Flow-matching U-Net image policy (third-person + wrist RGB, EE-local proprio, task-id
embedding). LIBERO-Spatial with LIBERO-PRO Spatial swap evaluation is the included test
case. Extracted from `traj-guided-policy` branch `feat/traj-il`
(`code_impl/mixed-diffuser`).

Module paths are kept as `diffusion_policy.*`, so checkpoints trained in the
original repo load here unchanged.

## Differences from Diffusion Policy (Chi et al., 2023)

- **Flow matching instead of DDPM.** `x_t = t·ε + (1−t)·a`, the network predicts the
  velocity `u = ε − a` with an MSE loss, `t ~ Beta(1.5, 1)` rescaled to `[0.001, 1]`
  (π0.5 convention: `t=1` noise, `t=0` action). Inference integrates 10 Euler steps from
  `t=1` to `t=0`; no noise scheduler. The `ConditionalUnet1D` backbone is unchanged and
  takes the continuous `t` as its timestep input.
- **Chunk-start EE-frame actions.** Per-step delta actions (xyz, axis-angle, gripper) are
  rotated into the end-effector frame at the start of each predicted chunk; all `H` steps
  share that frame. Translation and rotation are each projected onto the L2 unit ball, and
  actions are not normalized. The runner rotates them back to the controller frame.
- **EE-local proprio.** The proprio history (xyz, rotation matrix, gripper) is expressed in
  the latest EE frame, so the policy never sees the absolute EE pose. Only xyz and gripper
  are standardized.
- **Random third-person camera dropout.** During training, each sample's agentview frames
  (all observation steps) are zeroed with probability `policy.agentview_dropout` (default
  0.5); the wrist camera is never masked. `mask_agentview=True` runs the policy wrist-only
  at test time.
- **Categorical task conditioning.** Observations with `type: categorical` bypass the obs
  encoder and enter the U-Net global condition through a learned `nn.Embedding`, so task ids
  carry no artificial ordering.
- **Network and horizons.** `down_dims [128, 256]`, horizon 32, 2 observation steps,
  10 executed steps; ResNet18 from scratch with GroupNorm and ImageNet input normalization,
  one encoder per camera, random crop 128→112.
- **Checkpoint selection by rollout success** on training initial states, not by
  validation loss.

## Example: LIBERO-Spatial

### Data

External data, referenced by absolute path:
- Training demos: `imitation_learning/LIBERO/libero/datasets/spatial_4train/raw/<task>_demo_pc_instance_sg_50.hdf5`
  (`task.dataset.data_root`), 469 demos: the official 500 minus 31 demos with repeated grasps, dropped when these files were built.
- `data/spatial10_split.json`: per task, the `demo_<id>` ids held out for validation
  and test (5 + 5); training uses every other demo in the file (369/50/50).
- Evaluation scenes: LIBERO-PRO `libero_spatial_swap` BDDL and `.pruned_init` files under
  `imitation_learning/LIBERO-PRO/libero_data/` (`--pro-root`), 10 tasks × 50 initial states.

### Train

The default config is the agentview-dropout 0.5 recipe (301 epochs, rollout on
training initial states every 25 epochs, keep the top-1 checkpoint by rollout success):

```bash
PYTHONPATH=$PWD python train.py                                 # agentview dropout 0.5
PYTHONPATH=$PWD python train.py policy.agentview_dropout=0.0    # no dropout
```

Outputs go to `outputs/logs/<timestamp>/`.

### Evaluate on LIBERO-PRO Spatial swap

10 settle steps, then 220 policy steps, 10-step chunks, seed 2300:

```bash
PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/eval_fm_layouts.py \
    --checkpoint <ckpt> --name <run_name>
```

Writes `outputs/evaluations/<run_name>/results.json`.
