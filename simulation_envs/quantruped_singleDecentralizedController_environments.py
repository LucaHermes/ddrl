import numpy as np
from gym import spaces

from simulation_envs import QuantrupedMultiPoliciesEnv

class QuantrupedSingleControllerSuperEnv(QuantrupedMultiPoliciesEnv):
    """ Derived environment for control of the four-legged agent.
        Allows to instantiate multiple agents for control.
        
        Super class for all decentralized controller - control is split
        into four different, concurrent control units (policies) 
        each instantiated as a single agent. 
        
        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in derived classes and differs between the different architectures.
        - distribute_contact_cost: how to distribute (contact) costs individually to controllers 
        - concatenate_actions: how to integrate the control signals from the controllers
    """  
    policy_names = ['policy_legs']
    agent_names =  ["agent_FL","agent_HL","agent_HR","agent_FR"]

    def __init__(self, config):
        super().__init__(config)
        self.obs_indices = {
            "agent_FL" : self.env.get_obs_indices(['body', 'fl']), #[0,1,2,3,4, 5, 6,13,14,15,16,17,18,19,20,27,28,37,38]
            "agent_HL" : self.env.get_obs_indices(['body', 'hl']), #[0,1,2,3,4, 7, 8,13,14,15,16,17,18,21,22,29,30,39,40]
            "agent_HR" : self.env.get_obs_indices(['body', 'hr']), #[0,1,2,3,4, 9,10,13,14,15,16,17,18,23,24,31,32,41,42]
            "agent_FR" : self.env.get_obs_indices(['body', 'fr']), #[0,1,2,3,4,11,12,13,14,15,16,17,18,25,26,33,34,35,36]
        }
        self.action_indices = {
            'agent_FL' : self.env.get_action_indices(['fl']), #[2, 3]
            'agent_HL' : self.env.get_action_indices(['hl']), #[4, 5]
            'agent_HR' : self.env.get_action_indices(['hr']), #[6, 7],
            'agent_FR' : self.env.get_action_indices(['fr']), #[0, 1]
        }
        self.contact_force_indices = {
            'agent_FL' : self.env.get_contact_force_indices(['body', 'fl'], weights=[1./4., 1.]), #[2, 3]
            'agent_HL' : self.env.get_contact_force_indices(['body', 'hl'], weights=[1./4., 1.]), #[4, 5]
            'agent_HR' : self.env.get_contact_force_indices(['body', 'hr'], weights=[1./4., 1.]), #[6, 7],
            'agent_FR' : self.env.get_contact_force_indices(['body', 'fr'], weights=[1./4., 1.]), #[0, 1]
        }
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        return QuantrupedSingleControllerSuperEnv.policy_names[0]

    @staticmethod
    def return_policies(use_target_velocity=False):
        # For each agent the policy interface has to be defined.
        n_dims = 19 + use_target_velocity
        obs_space = spaces.Box(-np.inf, np.inf, (n_dims,), np.float64)
        policies = {
            QuantrupedSingleControllerSuperEnv.policy_names[0]: (None,
                obs_space, spaces.Box(-1., 1., (2,)), {}),
        }
        return policies

class QuantrupedSingleDecentralizedEnv(QuantrupedSingleControllerSuperEnv):
    
    def __init__(self, config):
        super().__init__(config)