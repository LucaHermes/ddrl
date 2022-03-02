import numpy as np
from gym import spaces

from simulation_envs import QuantrupedMultiPoliciesEnv

class QuantrupedFullyDecentralizedGlobalCostEnv(QuantrupedMultiPoliciesEnv):
    """ Derived environment for control of the four-legged agent.
        Uses four different, concurrent control units (policies) 
        each instantiated as a single agent. 
        
        Input scope of each controller: all available information. 
        
        Is used for the experiment on 
        S-II.2. Additional Experiment: Guiding training through local cost structure on flat terrain
        = using local costs, but global information.
        
        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
    """   
    
    # This is ordering of the policies as applied here:
    policy_names = ["policy_FL","policy_HL","policy_HR","policy_FR"]
    agent_names = ["agent_FL","agent_HL","agent_HR","agent_FR"]
    
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
        super().__init__(config)
        self.action_indices = {
            'agent_FL' : self.env.get_action_indices(['fl']), #[2, 3]
            'agent_HL' : self.env.get_action_indices(['hl']), #[4, 5]
            'agent_HR' : self.env.get_action_indices(['hr']), #[6, 7],
            'agent_FR' : self.env.get_action_indices(['fr']), #[0, 1]
        }
        self.obs_indices = {
            "agent_FL" : self.env.get_obs_indices(['body', 'fl']),
            "agent_HL" : self.env.get_obs_indices(['body', 'hl']),
            "agent_HR" : self.env.get_obs_indices(['body', 'hr']),
            "agent_FR" : self.env.get_obs_indices(['body', 'fr'])
        }
        self.contact_force_indices = {
            'agent_FL' : self.env.get_contact_force_indices(['body', 'fl'], weights=[1./4., 1.]), #[2, 3]
            'agent_HL' : self.env.get_contact_force_indices(['body', 'hl'], weights=[1./4., 1.]), #[4, 5]
            'agent_HR' : self.env.get_contact_force_indices(['body', 'hr'], weights=[1./4., 1.]), #[6, 7],
            'agent_FR' : self.env.get_contact_force_indices(['body', 'fr'], weights=[1./4., 1.]), #[0, 1]
        }

    def distribute_reward(self, reward_full, info, action_dict):
        fw_reward = info['reward_forward']
        rew = {}
        contact_costs = self.distribute_contact_cost()

        # Compute control costs:
        sum_control_cost = 0
        for policy_name in self.policy_names:
            sum_control_cost += np.sum(np.square(action_dict[policy_name]))
        #print("COSTs: ", sum_control_cost, " / ", action_dict)
        for policy_name in self.policy_names:
            rew[policy_name] = (fw_reward / len(self.policy_names)) \
                - (self.env.ctrl_cost_weight * 0.25 * sum_control_cost) \
                - (contact_costs[policy_name])
        return rew

    @staticmethod
    def return_policies(use_target_velocity=False):
        # For each agent the policy interface has to be defined.
        n_dims = 19 + use_target_velocity
        obs_space = spaces.Box(-np.inf, np.inf, (n_dims,), np.float64)
        policies = {
            QuantrupedFullyDecentralizedGlobalCostEnv.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            QuantrupedFullyDecentralizedGlobalCostEnv.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            QuantrupedFullyDecentralizedGlobalCostEnv.policy_names[2]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            QuantrupedFullyDecentralizedGlobalCostEnv.policy_names[3]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
        }
        return policies
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        if agent_id.startswith("agent_FL"):
            return "policy_FL"
        elif agent_id.startswith("agent_HL"):
            return "policy_HL"
        elif agent_id.startswith("agent_HR"):
            return "policy_HR"
        else:
            return "policy_FR" 
