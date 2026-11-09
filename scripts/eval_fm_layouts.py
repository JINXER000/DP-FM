"""Evaluate a flow-matching task expert on the LIBERO-PRO Spatial swap layouts.

For every task in the checkpoint's vocabulary, each initial state in
``<pro-root>/init_files/<suite>/<task>.pruned_init`` is one episode, with the scene from
``<pro-root>/bddl_files/<suite>/<task>.bddl``. Protocol: 10 uncounted settle steps, then at most
``--step-budget`` (220, LIBERO-PRO Spatial) policy steps; success is the BDDL goal.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from diffusion_policy.env_runner.libero_image_runner import LiberoEpisode, LiberoImageRunner
from diffusion_policy.workspace.train_flow_match_unet_image_workspace import TrainFlowMatchUnetImageWorkspace

ROOT = Path(__file__).resolve().parents[1]
PRO_ROOT = Path('/home/user/yzchen_ws/imitation_learning/LIBERO-PRO/libero_data')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--pro-root', type=Path, default=PRO_ROOT)
    parser.add_argument('--suite', default='libero_spatial_swap')
    parser.add_argument('--seed', type=int, default=2300)
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--step-budget', type=int, default=220)
    args = parser.parse_args()
    output = ROOT / 'outputs/evaluations' / args.name
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.resolve(strict=True)
    workspace = TrainFlowMatchUnetImageWorkspace.create_from_checkpoint(checkpoint)
    policy = workspace.ema_model.to('cuda:0').eval()
    vocabulary = list(workspace.cfg.task.task_names)
    episodes = []
    for task_name in vocabulary:
        init_states = args.pro_root / 'init_files' / args.suite / f'{task_name}.pruned_init'
        bddl_file = args.pro_root / 'bddl_files' / args.suite / f'{task_name}.bddl'
        for index in range(len(torch.load(init_states))):
            episodes.append(LiberoEpisode(task_name=task_name, init_state_index=index, sampling_seed=args.seed,
                                          scenario=f'{task_name}:{index}', bddl_file=str(bddl_file),
                                          init_states_path=str(init_states)))
    runner = LiberoImageRunner(output_dir=str(output), task_vocabulary=vocabulary, episodes=episodes,
                               num_envs=args.num_envs, prediction_horizon=policy.horizon,
                               n_action_steps=policy.n_action_steps, n_obs_steps=policy.n_obs_steps,
                               proprio_representation=policy.proprio_representation, max_steps=args.step_budget)
    result = runner.run(policy)
    per_task = {}
    for record in result['episodes']:
        row = per_task.setdefault(record['task_name'], Counter())
        row['episodes'] += 1
        row['successes'] += int(record['success'])
    summary = dict(checkpoint=str(checkpoint), suite=str(args.pro_root / args.suite), seed=args.seed,
                   protocol=dict(settle_steps=10, step_budget=args.step_budget,
                                 execution=f'{policy.n_action_steps}-step chunk, no replanning'),
                   successes=result['successes'], episodes=len(result['episodes']),
                   success_rate=result['success_rate'],
                   per_task={k: dict(v, success_rate=v['successes'] / v['episodes']) for k, v in per_task.items()},
                   episodes_detail=result['episodes'])
    (output / 'results.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'episodes_detail'}), flush=True)


if __name__ == '__main__':
    main()
