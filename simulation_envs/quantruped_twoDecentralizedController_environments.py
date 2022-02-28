import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces

from simulation_envs import QuantrupedMultiPoliciesEnv
        
class Quantruped_TwoSideControllers_Env(QuantrupedMultiPoliciesEnv):
    """ Derived environment for control of the four-legged agent.
        Uses two different, concurrent control units (policies) 
        each instantiated as a single agent. 
        
        Reward: If target_vel is given, is aiming for the given target velocity.
        
        There is one controller for each side of the agent.
        Input scope of each controller: 
        - two legs of that side.
        
        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_contact_cost: how to distribute (contact) costs individually to controllers 
    """  
    # This is ordering of the policies as applied here:
    policy_names = ["policy_LEFT","policy_RIGHT"]
    
    def __init__(self, config):
        self.obs_indices = {}
        # First global information: 
        # 0: height, 1-4: quaternion orientation torso
        # 5: hip FL angle, 6: knee FL angle
        # 7: hip HL angle, 8: knee HL angle
        # 9: hip HR angle, 10: knee HR angle
        # 11: hip FR angle, 12: knee FR angle
        # Velocities follow same ordering, but have in addition x and y vel.
        # 13-15: vel, 16-18: rotational velocity
        # 19: hip FL angle, 20: knee FL angle
        # 21: hip HL angle, 22: knee HL angle
        # 23: hip HR angle, 24: knee HR angle
        # 25: hip FR angle, 26: knee FR angle
        # Passive forces same ordering, only local information used
        # 27: hip FL angle, 28: knee FL angle
        # 29: hip HL angle, 30: knee HL angle
        # 31: hip HR angle, 32: knee HR angle
        # 33: hip FR angle, 34: knee FR angle
        # Last: control signals (clipped) from last time step
        # Unfortunately, different ordering (as the action spaces...)
        # 37: hip FL angle, 38: knee FL angle
        # 39: hip HL angle, 40: knee HL angle
        # 41: hip HR angle, 42: knee HR angle
        # 35: hip FR angle, 36: knee FR angle
        # Each controller only gets information from that body side: Left
        self.obs_indices["policy_LEFT"] =  [0,1,2,3,4, 5, 6, 7, 8,13,14,15,16,17,18,19,20,21,22,27,28,29,30,37,38,39,40]
        # Each controller only gets information from that body side: Right
        self.obs_indices["policy_RIGHT"] = [0,1,2,3,4, 9,10,11,12,13,14,15,16,17,18,23,24,25,26,31,32,33,34,41,42,35,36]
        # Each controller outputs four actions, below are the indices of the actions
        # in the action-list that gets passed to the environment.
        self.action_indices = {
            'policy_LEFT'  : [2, 3, 4, 5],
            'policy_RIGHT' : [6, 7, 0, 1]
        }
        super().__init__(config)

    def distribute_contact_cost(self):
        contact_cost = {}
        #print("CONTACT COST")
        #from mujoco_py import functions
        #functions.mj_rnePostConstraint(self.env.model, self.env.data)
        #print("From Ant Env: ", self.env.contact_cost)
        raw_contact_forces = self.env.sim.data.cfrc_ext
        contact_forces = np.clip(raw_contact_forces, -1., 1.)
        contact_costs = self.env.contact_cost_weight * np.square(contact_forces)
        global_contact_costs = np.sum(contact_costs[0:2])/4.
        contact_cost[self.policy_names[0]] = global_contact_costs + np.sum(contact_costs[2:5]) + np.sum(contact_costs[5:8])
        contact_cost[self.policy_names[1]] = global_contact_costs + np.sum(contact_costs[8:11]) + np.sum(contact_costs[11:])
        #print(contact_cost)
        #sum_c = 0.
        #for i in self.policy_names:
         #   sum_c += contact_cost[i]
        #print("Calculated: ", sum_c)
        return contact_cost
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        if agent_id.startswith("policy_LEFT"):
            return "policy_LEFT"
        else:
            return "policy_RIGHT" 
            
    @staticmethod
    def return_policies(obs_space):
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (27,), np.float64)
        policies = {
            Quantruped_TwoSideControllers_Env.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.,-1.]), np.array([+1.,+1.,+1.,+1.])), {}),
            Quantruped_TwoSideControllers_Env.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.,-1.,-1.]), np.array([+1.,+1.,+1.,+1.])), {})
        }
        return policies
        
class Quantruped_TwoDiagControllers_Env(QuantrupedMultiPoliciesEnv):
    """ Derived environment for control of the four-legged agent.
        Uses two different, concurrent control units (policies) 
        each instantiated as a single agent. 
        
        There is one controller for each pair of diagonal legs of the agent.
        Input scope of each controller: 
        - two diagonal legs
        
        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_contact_cost: how to distribute (contact) costs individually to controllers 
    """ 
    
    # This is ordering of the policies as applied here:
    policy_names = ["policy_FLHR","policy_HLFR"]
    
    def __init__(self, config):
        self.obs_indices = {}
        # First global information: 
        # 0: height, 1-4: quaternion orientation torso
        # 5: hip FL angle, 6: knee FL angle
        # 7: hip HL angle, 8: knee HL angle
        # 9: hip HR angle, 10: knee HR angle
        # 11: hip FR angle, 12: knee FR angle
        # Velocities follow same ordering, but have in addition x and y vel.
        # 13-15: vel, 16-18: rotational velocity
        # 19: hip FL angle, 20: knee FL angle
        # 21: hip HL angle, 22: knee HL angle
        # 23: hip HR angle, 24: knee HR angle
        # 25: hip FR angle, 26: knee FR angle
        # Passive forces same ordering, only local information used
        # 27: hip FL angle, 28: knee FL angle
        # 29: hip HL angle, 30: knee HL angle
        # 31: hip HR angle, 32: knee HR angle
        # 33: hip FR angle, 34: knee FR angle
        # Last: control signals (clipped) from last time step
        # Unfortunately, different ordering (as the action spaces...)
        # 37: hip FL angle, 38: knee FL angle
        # 39: hip HL angle, 40: knee HL angle
        # 41: hip HR angle, 42: knee HR angle
        # 35: hip FR angle, 36: knee FR angle
        # Each controller only gets information from two legs, diagonally arranged: FL-HR
        self.obs_indices["policy_FLHR"] = [0,1,2,3,4, 5, 6, 9,10,13,14,15,16,17,18,19,20,23,24,27,28,31,32,37,38,41,42]
        # Each controller only gets information from two legs, diagonally arranged: HL-FR
        self.obs_indices["policy_HLFR"] = [0,1,2,3,4, 7, 8,11,12,13,14,15,16,17,18,21,22,25,26,29,30,33,34,39,40,35,36]
        # Each controller outputs four actions, below are the indices of the actions
        # in the action-list that gets passed to the environment.
        self.action_indices = {
            'policy_FLHR' : [2, 3, 4, 5],
            'policy_HLFR' : [6, 7, 0, 1],
        }
        # TODO: Ask Malte, i think this one is correct
        self.action_indices = {
            'policy_FLHR' : [2, 3, 6, 7],
            'policy_HLFR' : [4, 5, 0, 1],
        }
        super().__init__(config)
        
    def distribute_contact_cost(self):
        contact_cost = {}
        #print("CONTACT COST")
        #from mujoco_py import functions
        #functions.mj_rnePostConstraint(self.env.model, self.env.data)
        #print("From Ant Env: ", self.env.contact_cost)
        raw_contact_forces = self.env.sim.data.cfrc_ext
        contact_forces = np.clip(raw_contact_forces, -1., 1.)
        contact_costs = self.env.contact_cost_weight * np.square(contact_forces)
        global_contact_costs = np.sum(contact_costs[0:2])/4.
        contact_cost[self.policy_names[0]] = global_contact_costs + np.sum(contact_costs[2:5]) + np.sum(contact_costs[8:11])
        contact_cost[self.policy_names[1]] = global_contact_costs + np.sum(contact_costs[5:8]) + np.sum(contact_costs[11:])
        #print(contact_cost)
        #sum_c = 0.
        #for i in self.policy_names:
         #   sum_c += contact_cost[i]
        #print("Calculated: ", sum_c)
        return contact_cost
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        if agent_id.startswith("policy_FLHR"):
            return "policy_FLHR"
        else:
            return "policy_HLFR" 
            
    @staticmethod
    def return_policies(use_target_velocity=False):
        # For each agent the policy interface has to be defined.
        n_dims = 27 + use_target_velocity
        obs_space = spaces.Box(-np.inf, np.inf, (n_dims,), np.float64)
        policies = {
            Quantruped_TwoDiagControllers_Env.policy_names[0]: (None,
                obs_space, spaces.Box(-1., +1, (4,)), {}),
            Quantruped_TwoDiagControllers_Env.policy_names[1]: (None,
                obs_space, spaces.Box(-1., +1, (4,)), {})
        }
        return policies