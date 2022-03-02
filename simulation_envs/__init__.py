from gym.envs.registration import registry, register, make, spec
from ray.tune.registry import register_env
from gym.wrappers.time_limit import TimeLimit
from simulation_envs.quantruped_v3 import QuAntrupedEnv, QuAntrupedTVelEnv
from simulation_envs.ant_v3_mujoco_2 import AntEnvMujoco2

# Importing the different multiagent environments.
from simulation_envs.quantruped_adaptor_multi_environment import QuantrupedMultiPoliciesEnv
from simulation_envs.quantruped_GraphDecentralizedController_environments import QuantrupedDecentralizedGraphEnv
from simulation_envs.quantruped_fourDecentralizedController_environments import QuantrupedFullyDecentralizedEnv
from simulation_envs.quantruped_fourDecentralizedController_environments import Quantruped_LocalSingleNeighboringLeg_Env
from simulation_envs.quantruped_fourDecentralizedController_environments import Quantruped_LocalSingleDiagonalLeg_Env
from simulation_envs.quantruped_fourDecentralizedController_environments import Quantruped_Local_Env
from simulation_envs.quantruped_fourDecentralizedController_environments import Quantruped_LocalSingleToFront_Env
from simulation_envs.quantruped_twoDecentralizedController_environments import Quantruped_TwoSideControllers_Env
from simulation_envs.quantruped_twoDecentralizedController_environments import Quantruped_TwoDiagControllers_Env

from simulation_envs.quantruped_fourDecentralizedController_GlobalCosts_environments import QuantrupedFullyDecentralizedGlobalCostEnv

from simulation_envs.quantruped_singleDecentralizedController_environments import QuantrupedSingleDecentralizedEnv

# Register Gym environment. 
register(
	id='QuAntruped-v3',
	entry_point='simulation_envs.quantruped_v3:QuAntrupedEnv',
	max_episode_steps=1000,
	reward_threshold=6000.0,
)
register(
	id='QuAntrupedTVel-v3',
	entry_point='simulation_envs.quantruped_v3:QuAntrupedTVelEnv',
	max_episode_steps=1000,
	reward_threshold=1000.0,
)

register(
	id='Ant_Muj2-v3',
	entry_point='simulation_envs.ant_v3_mujoco_2:AntEnvMujoco2',
	max_episode_steps=1000,
	reward_threshold=6000.0,
)

# Register single agent ray environment (wrapping gym environment).
register_env("Ant_Muj2-v3", lambda config: TimeLimit(AntEnvMujoco2(), max_episode_steps=1000))
register_env("QuAntruped-v3", lambda config: TimeLimit(QuAntrupedEnv(), max_episode_steps=1000))
register_env("QuAntrupedTVel-v3", lambda config: TimeLimit(QuAntrupedTVelEnv(), max_episode_steps=1000))

# Register multiagent environments (allowing individual access to individual legs).
register_env("QuantrupedMultiEnv_Centralized", lambda config: QuantrupedMultiPoliciesEnv(config) )
register_env("QuantrupedMultiEnv_Decentral_Graph", lambda config: QuantrupedDecentralizedGraphEnv(config) )
register_env("QuantrupedMultiEnv_FullyDecentral", lambda config: QuantrupedFullyDecentralizedEnv(config) )
register_env("QuantrupedMultiEnv_FullyDecentralGlobalCost", lambda config: QuantrupedFullyDecentralizedGlobalCostEnv(config) )
register_env("QuantrupedMultiEnv_SingleNeighbor", lambda config: Quantruped_LocalSingleNeighboringLeg_Env(config) )
register_env("QuantrupedMultiEnv_SingleDiagonal", lambda config: Quantruped_LocalSingleDiagonalLeg_Env(config) )
register_env("QuantrupedMultiEnv_SingleToFront", lambda config: Quantruped_LocalSingleToFront_Env(config) )
register_env("QuantrupedMultiEnv_Local", lambda config: Quantruped_Local_Env(config) )
register_env("QuantrupedMultiEnv_TwoSides", lambda config: Quantruped_TwoSideControllers_Env(config) )
register_env("QuantrupedMultiEnv_TwoDiags", lambda config: Quantruped_TwoDiagControllers_Env(config) )

register_env('QuantrupedMultiEnv_SharedDecentral', lambda config: QuantrupedSingleDecentralizedEnv(config) )
