import numpy as np
from gym import spaces

from simulation_envs import QuantrupedMultiPoliciesEnv

class QuantrupedFourControllerSuperEnv(QuantrupedMultiPoliciesEnv):
    """ Derived environment for control of the four-legged agent.
        Allows to instantiate multiple agents for control.
        
        Super class for all decentralized controller - control is split
        into four different, concurrent control units (policies) 
        each instantiated as a single agent. 
        
        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_contact_cost: how to distribute (contact) costs individually to controllers 
    """  
    def __init__(self, config):
        super().__init__(config)
        self.action_indices = {
            'policy_FL' : self.env.get_action_indices(['fl']), #[2, 3]
            'policy_HL' : self.env.get_action_indices(['hl']), #[4, 5]
            'policy_HR' : self.env.get_action_indices(['hr']), #[6, 7],
            'policy_FR' : self.env.get_action_indices(['fr']), #[0, 1]
        }
        self.contact_force_indices = {
            'policy_FL' : self.env.get_contact_force_indices(['body', 'fl'], weights=[1./4., 1.]), #[2, 3]
            'policy_HL' : self.env.get_contact_force_indices(['body', 'hl'], weights=[1./4., 1.]), #[4, 5]
            'policy_HR' : self.env.get_contact_force_indices(['body', 'hr'], weights=[1./4., 1.]), #[6, 7],
            'policy_FR' : self.env.get_contact_force_indices(['body', 'fr'], weights=[1./4., 1.]), #[0, 1]
        }
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        if agent_id.startswith("policy_FL"):
            return "policy_FL"
        elif agent_id.startswith("policy_HL"):
            return "policy_HL"
        elif agent_id.startswith("policy_HR"):
            return "policy_HR"
        else:
            return "policy_FR" 

class QuantrupedFullyDecentralizedEnv(QuantrupedFourControllerSuperEnv):
    """ Derived environment for control of the four-legged agent.
        Uses four different, concurrent control units (policies) 
        each instantiated as a single agent. 
        
        Input scope of each controller: only the controlled leg.
        
        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
    """ 
    
    # This is ordering of the policies as applied here:
    policy_names = ["policy_FL","policy_HL","policy_HR","policy_FR"]
    
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
        super().__init__(config)
        self.obs_indices["policy_FL"] = self.env.get_obs_indices(['body', 'fl']) #[0,1,2,3,4, 5, 6,13,14,15,16,17,18,19,20,27,28,37,38]
        self.obs_indices["policy_HL"] = self.env.get_obs_indices(['body', 'hl']) #[0,1,2,3,4, 7, 8,13,14,15,16,17,18,21,22,29,30,39,40]
        self.obs_indices["policy_HR"] = self.env.get_obs_indices(['body', 'hr']) #[0,1,2,3,4, 9,10,13,14,15,16,17,18,23,24,31,32,41,42]
        self.obs_indices["policy_FR"] = self.env.get_obs_indices(['body', 'fr']) #[0,1,2,3,4,11,12,13,14,15,16,17,18,25,26,33,34,35,36]

    @staticmethod
    def return_policies(use_target_velocity=False):
        # For each agent the policy interface has to be defined.
        n_dims = 19 + use_target_velocity
        obs_space = spaces.Box(-np.inf, np.inf, (n_dims,), np.float64)
        policies = {
            QuantrupedFullyDecentralizedEnv.policy_names[0]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
            QuantrupedFullyDecentralizedEnv.policy_names[1]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
            QuantrupedFullyDecentralizedEnv.policy_names[2]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
            QuantrupedFullyDecentralizedEnv.policy_names[3]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
        }
        return policies
        
class Quantruped_LocalSingleNeighboringLeg_Env(QuantrupedFourControllerSuperEnv):
    """ Derived environment for control of the four-legged agent.
        Uses four different, concurrent control units (policies) 
        each instantiated as a single agent. 
        
        Input scope of each controller: 
        - controlled leg
        - plus from an additional neighboring leg (counterclockwise)
        
        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
    """ 
    
    # This is ordering of the policies as applied here:
    policy_names = ["policy_FL","policy_HL","policy_HR","policy_FR"]
    
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
        super().__init__(config) 
        # FL also gets local information from HL
        self.obs_indices["policy_FL"] = self.env.get_obs_indices(['body', 'fl', 'hl'])
        # HL also gets local information from HR
        self.obs_indices["policy_HL"] = self.env.get_obs_indices(['body', 'hl', 'hr'])
        # HR also gets local information from FR
        self.obs_indices["policy_HR"] = self.env.get_obs_indices(['body', 'hr', 'fr'])
        # FR also gets local information from FL
        self.obs_indices["policy_FR"] = self.env.get_obs_indices(['body', 'fr', 'fl'])

        
    @staticmethod
    def return_policies(use_target_velocity=False):
        # For each agent the policy interface has to be defined.
        n_dims = 27 + use_target_velocity
        obs_space = spaces.Box(-np.inf, np.inf, (n_dims,), np.float64)
        policies = {
            Quantruped_LocalSingleNeighboringLeg_Env.policy_names[0]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
            Quantruped_LocalSingleNeighboringLeg_Env.policy_names[1]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
            Quantruped_LocalSingleNeighboringLeg_Env.policy_names[2]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
            Quantruped_LocalSingleNeighboringLeg_Env.policy_names[3]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
        }
        return policies

class Quantruped_LocalSingleDiagonalLeg_Env(QuantrupedFourControllerSuperEnv):
    """ Derived environment for control of the four-legged agent.
        Uses four different, concurrent control units (policies) 
        each instantiated as a single agent. 
        
        Input scope of each controller: 
        - controlled leg
        - plus from the diagonal leg
        
        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
    """ 
    
    # This is ordering of the policies as applied here:
    policy_names = ["policy_FL","policy_HL","policy_HR","policy_FR"]
    
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
        super().__init__(config)
        # FL also gets local information from HR
        self.obs_indices["policy_FL"] = self.env.get_obs_indices(['body', 'fl', 'hr'])
        # HL also gets local information from FR
        self.obs_indices["policy_HL"] = self.env.get_obs_indices(['body', 'hl', 'fr'])
        # HR gets local information like FL
        self.obs_indices["policy_HR"] = self.obs_indices["policy_FL"]
        # FR gets local information from HL
        self.obs_indices["policy_FR"] = self.obs_indices["policy_HL"]
            
    @staticmethod
    def return_policies(use_target_velocity=False):
        # For each agent the policy interface has to be defined.
        n_dims = 27 + use_target_velocity
        obs_space = spaces.Box(-np.inf, np.inf, (n_dims,), np.float64)
        policies = {
            Quantruped_LocalSingleDiagonalLeg_Env.policy_names[0]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
            Quantruped_LocalSingleDiagonalLeg_Env.policy_names[1]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
            Quantruped_LocalSingleDiagonalLeg_Env.policy_names[2]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
            Quantruped_LocalSingleDiagonalLeg_Env.policy_names[3]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
        }
        return policies
        
class Quantruped_LocalSingleToFront_Env(QuantrupedFourControllerSuperEnv):
    """ Derived environment for control of the four-legged agent.
        Uses four different, concurrent control units (policies) 
        each instantiated as a single agent. 
        
        Input scope of each controller: 
        - controlled leg
        - plus from an additional neighboring leg:
            for front legs from hind legs
            for hind legs from other hind leg
        
        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
    """    
    
    # This is ordering of the policies as applied here:
    policy_names = ["policy_FL","policy_HL","policy_HR","policy_FR"]
    
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
        super().__init__(config)
        # FL also gets local information from HL (towards front)
        self.obs_indices["policy_FL"] = self.env.get_obs_indices(['body', 'fl', 'hl'])
        # HL also gets local information from HR (from side at back)
        self.obs_indices["policy_HL"] = self.env.get_obs_indices(['body', 'hl', 'hr'])
        # HR also gets local information from HL (from side at back)
        self.obs_indices["policy_HR"] = self.env.get_obs_indices(['body', 'hr', 'hl'])
        # FR also gets local information from HR (towards front)
        self.obs_indices["policy_FR"] = self.env.get_obs_indices(['body', 'fr', 'hr'])
    
    @staticmethod
    def return_policies(use_target_velocity=False):
        # For each agent the policy interface has to be defined.
        n_dims = 27 + use_target_velocity
        obs_space = spaces.Box(-np.inf, np.inf, (n_dims,), np.float64)
        policies = {
            Quantruped_LocalSingleToFront_Env.policy_names[0]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
            Quantruped_LocalSingleToFront_Env.policy_names[1]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
            Quantruped_LocalSingleToFront_Env.policy_names[2]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
            Quantruped_LocalSingleToFront_Env.policy_names[3]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
        }
        return policies
        
class Quantruped_Local_Env(QuantrupedFourControllerSuperEnv):
    """ Derived environment for control of the four-legged agent.
        Uses four different, concurrent control units (policies) 
        each instantiated as a single agent. 
        
        Input scope of each controller: 
        - controlled leg
        - plus from both neighboring legs
        
        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
    """    
    
    # This is ordering of the policies as applied here:
    policy_names = ["policy_FL","policy_HL","policy_HR","policy_FR"]
    
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
        super().__init__(config)
        # FL also gets local information from HL and FR
        self.obs_indices["policy_FL"] = self.env.get_obs_indices(['body', 'fl', 'hl', 'fr'])
        # HL also gets local information from HR and FL
        self.obs_indices["policy_HL"] = self.env.get_obs_indices(['body', 'hl', 'hr', 'fl'])
        # HR also gets local information from FR and HL
        self.obs_indices["policy_HR"] = self.env.get_obs_indices(['body', 'hr', 'fr', 'hl'])
        # FR also gets local information from FL and HR
        self.obs_indices["policy_FR"] = self.env.get_obs_indices(['body', 'fr', 'fl', 'hr'])
            
    @staticmethod
    def return_policies(use_target_velocity=False):
        # For each agent the policy interface has to be defined.
        n_dims = 35 + use_target_velocity
        obs_space = spaces.Box(-np.inf, np.inf, (n_dims,), np.float64)
        policies = {
            Quantruped_Local_Env.policy_names[0]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
            Quantruped_Local_Env.policy_names[1]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
            Quantruped_Local_Env.policy_names[2]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
            Quantruped_Local_Env.policy_names[3]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
        }
        return policies
