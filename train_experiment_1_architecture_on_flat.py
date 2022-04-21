import simulation_envs
import numpy as np
import gym
import uuid
from gym import spaces

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import pretty_print
from ray import tune
from ray.tune import grid_search
import time

import models

import argparse

# Switch between different approaches.
parser = argparse.ArgumentParser()
parser.add_argument("--policy_scope", required=False)
parser.add_argument("--model", required=False, default="ffn")
parser.add_argument("--name", required=False, default=None)
parser.add_argument("--norm_reward", action='store_true', default=False)
parser.add_argument("--global_reward", action='store_true', default=False)
parser.add_argument("--target_velocity", required=False)
args = parser.parse_args()

use_target_velocity = 'target_velocity' in args and args.target_velocity

# currently norm_reward, global_reward and use_target_velocity are mutually exclusive (should we make this independent?)
assert not (args.norm_reward and args.global_reward)
assert not (args.norm_reward and use_target_velocity)
assert not (args.global_reward and use_target_velocity)

# Possible values: 
#   QuantrupedMultiEnv_Centralized - single controller, global information
#   QuantrupedMultiEnv_FullyDecentral - four decentralized controlller, information 
#       from the controlled leg only
#   QuantrupedMultiEnv_SingleNeighbor - four decentralized controlller, information 
#       from the controlled leg plus neighbor (ccw)
#   QuantrupedMultiEnv_SingleDiagonal - four decentralized controlller, information 
#       from the controlled leg plus diagonal
#   QuantrupedMultiEnv_SingleToFront - four decentralized controlller, information 
#       from the controlled leg plus one neighbor, for front legs from hind legs
#       for hind legs, the other hind leg
#   QuantrupedMultiEnv_Local - four decentralized controlller, information 
#       from the controlled leg plus both neighboring legs
#   QuantrupedMultiEnv_TwoSides - two decentralized controlller, one for each side, 
#       information from the controlled legs 
#   QuantrupedMultiEnv_TwoDiags - two decentralized controlller, controlling a pair of 
#       diagonal legs, 
#       information from the controlled legs 
#   QuantrupedMultiEnv_FullyDecentralGlobalCost - four decentralized controlller, information 
#       from the controlled leg; variation: global costs are used.

if 'policy_scope' in args and args.policy_scope: 
    policy_scope = args.policy_scope
else:
    policy_scope = 'QuantrupedMultiEnv_Centralized'
 
if policy_scope=="QuantrupedMultiEnv_FullyDecentral":
    from simulation_envs.quantruped_fourDecentralizedController_environments import QuantrupedFullyDecentralizedEnv as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_Decentral_Graph":
    from simulation_envs.quantruped_GraphDecentralizedController_environments import QuantrupedDecentralizedGraphEnv as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_DecentralShared_Graph":
    from simulation_envs.quantruped_GraphDecentralizedController_environments import QuantrupedDecentralizedSharedGraphEnv as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_SingleNeighbor":
    from simulation_envs.quantruped_fourDecentralizedController_environments import Quantruped_LocalSingleNeighboringLeg_Env as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_SingleDiagonal":
    from simulation_envs.quantruped_fourDecentralizedController_environments import Quantruped_LocalSingleDiagonalLeg_Env as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_SingleToFront":
    from simulation_envs.quantruped_fourDecentralizedController_environments import Quantruped_LocalSingleToFront_Env as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_Local":
    from simulation_envs.quantruped_fourDecentralizedController_environments import Quantruped_Local_Env as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_TwoSides":
    from simulation_envs.quantruped_twoDecentralizedController_environments import Quantruped_TwoSideControllers_Env as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_TwoDiags":
    from simulation_envs.quantruped_twoDecentralizedController_environments import Quantruped_TwoDiagControllers_Env as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_FullyDecentralGlobalCost":
    from simulation_envs.quantruped_fourDecentralizedController_GlobalCosts_environments import QuantrupedFullyDecentralizedGlobalCostEnv as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_SharedDecentral":
    from simulation_envs.quantruped_singleDecentralizedController_environments import QuantrupedSingleDecentralizedEnv as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_SharedDecentralLegID":
    from simulation_envs.quantruped_singleDecentralizedController_environments import QuantrupedSingleDecentralizedLegIDEnv as QuantrupedEnv
elif policy_scope=="QuantrupedMultiEnv_SharedDecentralLegTransforms":
    from simulation_envs.quantruped_singleDecentralizedController_environments import QuantrupedSingleDecentralizedLegTransforms as QuantrupedEnv
else:
    from simulation_envs.quantruped_centralizedController_environment import Quantruped_Centralized_Env as QuantrupedEnv

# Init ray: First line on server, second for laptop
#ray.init(num_cpus=30, ignore_reinit_error=True)
ray.init(num_gpus=1, ignore_reinit_error=True)

config = ppo.DEFAULT_CONFIG.copy()
#print(config)
#asd
config['env'] = policy_scope
print("SELECTED ENVIRONMENT: ", policy_scope, " = ", QuantrupedEnv)

#gpu_count = 1. / 10.
#num_gpus = gpu_count / 3.
#num_gpus_per_worker = (gpu_count - num_gpus) / 1.

#config['num_gpus']=1#num_gpus
config['num_workers']=2
config['num_envs_per_worker']=4
#config['num_gpus_per_worker']=1 #num_gpus_per_worker
# Have to disable env checking becaus our environments are not compatible with
# empty actions dicts
config['disable_env_checking'] = True
#config['nump_gpus']=1

# used grid_search([4000, 16000, 65536], didn't matter too much
config['train_batch_size'] = 16000

# Baseline Defaults:
config['gamma'] = 0.99
config['lambda'] = 0.95 
       
config['entropy_coeff'] = 0. # again used grid_search([0., 0.01]) for diff. values from lit.
config['clip_param'] = 0.2

config['vf_loss_coeff'] = 0.5
#config['vf_clip_param'] = 4000.

config['observation_filter'] = 'NoFilter'
# this is necessary, because otherwise the observation is flattened
# (even in the "obs" entry of the input dict)
config['_disable_preprocessor_api'] = True

config['sgd_minibatch_size'] = 128
config['num_sgd_iter'] = 10
config['lr'] = 3e-4
config['grad_clip']=0.5

config['model']['custom_model'] = args.model #"fc_glorot_uniform_init"
config['model']['fcnet_hiddens'] = [64, 64]

#config['seed'] = round(time.time())

# For running tune, we have to provide information on 
# the multiagent which are part of the MultiEnvs
policies = QuantrupedEnv.return_policies(use_target_velocity=False)

config["multiagent"] = {
        "policies": policies,
        "policy_mapping_fn": QuantrupedEnv.policy_mapping_fn,
        "policies_to_train": QuantrupedEnv.policy_names, #, "dec_B_policy"],
    }

config['env_config']['ctrl_cost_weight'] = 0.5#grid_search([5e-4,5e-3,5e-2])
config['env_config']['contact_cost_weight'] =  5e-2 #grid_search([5e-4,5e-3,5e-2])
config['env_config']['norm_reward'] = args.norm_reward
config['env_config']['global_reward'] = args.global_reward

# Parameters for defining environment:
# Heightfield smoothness (between 0.6 and 1.0 are OK)
config['env_config']['hf_smoothness'] = 1.0
# Defining curriculum learning
config['env_config']['curriculum_learning'] =  False
config['env_config']['range_smoothness'] =  [1., 0.6]
config['env_config']['range_last_timestep'] =  10000000

# Setting target velocity (range of up to 2.)
if use_target_velocity: 
    config['env_config']['target_velocity'] = float(args.target_velocity)

# For curriculum learning: environment has to be updated every epoch
def on_train_result(info):
    result = info["result"]
    trainer = info["trainer"]
    timesteps_res = result["timesteps_total"]
    trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env( lambda env: env.update_environment_after_epoch( timesteps_res ) )) 

config["callbacks"] = { "on_train_result" : on_train_result }
config['logger_config'] = {
    "wandb": {
        "project": "DDRL",
        "group"  : args.name + '_' + str(uuid.uuid4())
    }
}

if args.name:
    policy_scope = f'{policy_scope}:{args.name}'

run_prefix = 'HF_10_'

if args.norm_reward:
    run_prefix = 'NormRew_'
if args.global_reward:
    run_prefix = 'GR_'
if use_target_velocity:
    run_prefix = 'Tvel_'

# Call tune and run (for evaluation: 10 seeds up to 20M steps; only centralized controller
# required that much of time; decentralized controller should show very good results 
# after 5M steps.
analysis = tune.run(
    "PPO",
    name=(run_prefix + policy_scope),
    num_samples=10,
    checkpoint_at_end=True,
    checkpoint_freq=312,
    stop={"timesteps_total": 20000000},
    #resources_per_trial={ "cpu" : 2, "gpu" : 1. },
    config=config,
    loggers=[WandbLogger]
)
