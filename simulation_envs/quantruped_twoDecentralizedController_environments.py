import numpy as np
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
    agent_names = ["agent_LEFT","agent_RIGHT"]
    
    def __init__(self, config):
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
        #self.obs_indices["policy_LEFT"] =  [0,1,2,3,4, 5, 6, 7, 8,13,14,15,16,17,18,19,20,21,22,27,28,29,30,37,38,39,40]
        # Each controller only gets information from that body side: Right
        #self.obs_indices["policy_RIGHT"] = [0,1,2,3,4, 9,10,11,12,13,14,15,16,17,18,23,24,25,26,31,32,33,34,41,42,35,36]
        super().__init__(config)
        # Each controller outputs four actions, below are the indices of the actions
        # in the action-list that gets passed to the environment.
        self.action_indices = {
            'agent_LEFT'  : self.env.get_action_indices(['fl', 'hl']), #[2, 3, 4, 5],
            'agent_RIGHT' : self.env.get_action_indices(['hr', 'fr']), #[6, 7, 0, 1],
        }
        self.obs_indices = {
            # Each controller only gets information from that body side: Left
            "agent_LEFT"  : self.env.get_obs_indices(['body', 'fl', 'hl']),
            # Each controller only gets information from that body side: Right
            "agent_RIGHT" : self.env.get_obs_indices(['body', 'hr', 'fr'])
        }
        self.contact_force_indices = {
            'agent_LEFT'  : self.env.get_contact_force_indices(['body', 'fl', 'hl'], weights=[1./2., 1., 1.]), #[2, 3]
            'agent_RIGHT' : self.env.get_contact_force_indices(['body', 'hr', 'fr'], weights=[1./2., 1., 1.]), #[4, 5]
        }
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        if agent_id.startswith("agent_LEFT"):
            return "policy_LEFT"
        else:
            return "policy_RIGHT" 
            
    @staticmethod
    def return_policies(use_target_velocity=False):
        # For each agent the policy interface has to be defined.
        n_dims = 27 + use_target_velocity
        obs_space = spaces.Box(-np.inf, np.inf, (n_dims,), np.float64)
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
    agent_names = ["agent_FLHR","agent_HLFR"]
    
    def __init__(self, config):
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
        # Each controller outputs four actions, below are the indices of the actions
        # in the action-list that gets passed to the environment.
        super().__init__(config)
        # TODO: Ask Malte, i think this one is correct
        self.action_indices = {
            'agent_FLHR' : self.env.get_action_indices(['fl', 'hr']), #[2, 3, 6, 7],
            'agent_HLFR' : self.env.get_action_indices(['hl', 'fr']), #[4, 5, 0, 1],
        }
        self.obs_indices = {
            # Each controller only gets information from two legs, diagonally arranged: FL-HR
            "agent_FLHR" : self.env.get_obs_indices(['body', 'fl', 'hr']),
            # Each controller only gets information from two legs, diagonally arranged: HL-FR
            "agent_HLFR" : self.env.get_obs_indices(['body', 'hl', 'fr'])
        }
        self.contact_force_indices = {
            'agent_FLHR' : self.env.get_contact_force_indices(['body', 'fl', 'hr'], weights=[1./2., 1., 1.]), #[2, 3]
            'agent_HLFR' : self.env.get_contact_force_indices(['body', 'hl', 'fr'], weights=[1./2., 1., 1.]), #[4, 5]
        }
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        if agent_id.startswith("agent_FLHR"):
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