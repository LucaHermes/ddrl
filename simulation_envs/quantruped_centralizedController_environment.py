import numpy as np
from gym import spaces

from simulation_envs import QuantrupedMultiPoliciesEnv

class Quantruped_Centralized_Env(QuantrupedMultiPoliciesEnv):
    """ Derived environment for control of the four-legged agent.
        Allows to instantiate multiple agents for control.
        
        Centralized approach: Single agent (as standard DRL approach)
        controls all degrees of freedom of the agent.
        
        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_contact_cost: how to distribute (contact) costs individually to controllers 
    """  
    
    # This is ordering of the policies as applied here:
    policy_names = ["central_policy"]
    agent_names = ["central_policy"]
    
    def __init__(self, config):
        super().__init__(config)
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
        # The central policy gets all observations
        self.obs_indices = {
            "central_agent" : self.env.get_obs_indices() # all observations
        }
        self.action_indices = {
            "central_agent" : self.env.get_action_indices()
        }
        self.contact_force_indices = {
            "central_agent" : self.env.get_contact_force_indices()
        }
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        return Quantruped_Centralized_Env.policy_names[0]
            
    @staticmethod
    def return_policies(use_target_velocity=False):
        n_dims = 43 + use_target_velocity
        obs_space = spaces.Box(-np.inf, np.inf, (n_dims,), np.float64)
        # For each agent the policy interface has to be defined.
        policies = {
            QuantrupedMultiPoliciesEnv.policy_names[0]: (
                None,
                obs_space, 
                spaces.Box(-1., +1., (8,)), 
                {})
        }
        return policies