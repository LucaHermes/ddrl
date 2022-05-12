import os
import sys
sys.path.append(os.getcwd())
import ray
import pickle as pickle
import argparse
import re
import pathlib
from pprint import pprint

from ray.tune.registry import get_trainable_cls
from ray.rllib.evaluation.worker_set import WorkerSet

import simulation_envs
import models
from evaluation.rollout_episodes import rollout_episodes
import time

ray.init()

"""
    Visualizing a learned (multiagent) controller,
    for evaluation or visualisation.
    
    This is adapted from rllib's rollout.py
    (github.com/ray/rllib/rollout.py)
"""
parser = argparse.ArgumentParser(
    description='Run the policy and create a video.'
)
parser.add_argument('--path', '-p', type=str, required=True, 
    help='Path to the log files from training, e.g. ~/ray_results/runname/trialname.')
parser.add_argument('--epoch', '-e', type=int, required=False,
    help='''Epoch of the checkpoint. This depends on the log interval,'''
         '''a checkpoint might not exist for every epoch.'''
         '''Defaults to the latest epoch for which a checkpoint is present.''')
parser.add_argument('--steps', '-s', type=int, required=False, default=400,
    help='Number of steps that the agent is run, defaults to 400.')
parser.add_argument('--out', '-o', type=str, required=False, 
    help='Filename of the video with or without path.')


args = parser.parse_args()

path = args.path
epoch = args.epoch
out_file = args.out
num_steps = args.steps
out_dir = os.path.dirname(out_file)
out_file = os.path.basename(out_file)
num_episodes = 1

has_epoch = epoch is not None
ckpt_pattern = '*_0*' + (str(epoch) if has_epoch else '')
ckpts = pathlib.Path(path).rglob(ckpt_pattern)
ckpt_path = sorted(ckpts, reverse=not has_epoch)[0]

if not has_epoch:
    epoch = int(str(ckpt_path).split('_')[-1].lstrip('0'))

ckpt_file = os.path.join(ckpt_path, f'checkpoint-{epoch}')
config_file = os.path.join(ckpt_path, "..", "params.pkl")

if not out_dir:
    out_dir = os.path.join('videos',
        config_file.partition('MultiEnv_')[2].partition('/')[0])


frames_dir = os.path.join(out_dir, "frames")

print('\n\n' + '='*80)
print('Found checkpoint of epoch', epoch)
print('Checkpoint file:', ckpt_file)


os.path.isdir(frames_dir) or os.makedirs(frames_dir)
os.path.isdir(out_dir) or os.makedirs(out_dir)
out_file = os.path.join(out_dir, out_file)

# Afterwards put together using
# ffmpeg -framerate 20 -pattern_type glob -i '*.png' -filter:v scale=720:-1 -vcodec libx264 -pix_fmt yuv420p -g 1 out.mp4

#HF_10_QuantrupedMultiEnv_Local/PPO_QuantrupedMultiEnv_Local_1a49c_00003_3_2020-12-04_12-08-57
#HF_10_QuantrupedMultiEnv_TwoSides/PPO_QuantrupedMultiEnv_TwoSides_6654b_00006_6_2020-12-06_17-42-00
#HF_10_QuantrupedMultiEnv_FullyDecentral/PPO_QuantrupedMultiEnv_FullyDecentral_19697_00004_4_2020-12-04_12-08-56

with open(config_file, "rb") as f:
    config = pickle.load(f)

print('Config:')
pprint(config)

print('\n\n' + '='*80)

# Starting ray and setting up ray.
if "num_workers" in config:
    config["num_workers"] = min(2, config["num_workers"])

cls = get_trainable_cls('PPO')

# Setting config values (required for compatibility between versions)
config["create_env_on_driver"] = True
#config['env_config']['hf_smoothness'] = smoothness
if "no_eager_on_workers" in config:
    del config["no_eager_on_workers"]

# Load state from checkpoint.
agent = cls(env=config['env'], config=config)
agent.restore(ckpt_file)

# Retrieve environment for the trained agent.
if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
    env = agent.workers.local_worker().env

time.sleep(2)

# Rolling out simulation = stepping through simulation. 
rollout_episodes(env, agent, 
    num_episodes=num_episodes, 
    num_steps=num_steps, 
    render=True,
    save_images=os.path.join(frames_dir, "img_"), 
    save_obs=out_dir)

agent.stop()
