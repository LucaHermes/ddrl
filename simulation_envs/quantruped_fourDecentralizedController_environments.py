import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.filter import MeanStdFilter
import numpy as np
import mujoco_py
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
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in derived classes and differs between the different architectures.
        - distribute_contact_cost: how to distribute (contact) costs individually to controllers 
        - concatenate_actions: how to integrate the control signals from the controllers
    """  

    def distribute_observations(self, obs_full):
        """ 
        Construct dictionary that routes to each policy only the relevant
        local information.
        """
        obs_distributed = {}
        for policy_name in self.policy_names:
            obs_distributed[policy_name] = obs_full[self.obs_indices[policy_name],]
        return obs_distributed

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
        contact_cost[self.policy_names[0]] = global_contact_costs + np.sum(contact_costs[2:5])
        contact_cost[self.policy_names[1]] = global_contact_costs + np.sum(contact_costs[5:8])
        contact_cost[self.policy_names[2]] = global_contact_costs + np.sum(contact_costs[8:11])
        contact_cost[self.policy_names[3]] = global_contact_costs + np.sum(contact_costs[11:])
        #print(contact_cost)
        #sum_c = 0.
        #for i in self.policy_names:
         #   sum_c += contact_cost[i]
        #print("Calculated: ", sum_c)
        return contact_cost
        
    def concatenate_actions(self, action_dict):
        # Return actions in the (DIFFERENT in Mujoco) order FR - FL - HL - HR
        actions = np.concatenate( (action_dict[self.policy_names[3]],
            action_dict[self.policy_names[0]],
            action_dict[self.policy_names[1]],
            action_dict[self.policy_names[2]]) )
        return actions
        
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

class QuantrupedDecentralizedGraphEnv(QuantrupedFourControllerSuperEnv):
    """ Derived environment for control of the four-legged agent.
        Uses four different, concurrent control units (policies) 
        each instantiated as a single agent. 

        Input scope of each controller: only the controlled leg.

        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in the obs_indices for each leg.
    """ 
    
    # This is ordering of the policies as applied here:
    policy_names = ["policy_FL","policy_HL","policy_HR","policy_FR"]
    #obs_indices = {
    #    "policy_FL" : [0,1,2,3,4, 5, 6,13,14,15,16,17,18,19,20,27,28,37,38],
    #    "policy_HL" : [0,1,2,3,4, 7, 8,13,14,15,16,17,18,21,22,29,30,39,40],
    #    "policy_HR" : [0,1,2,3,4, 9,10,13,14,15,16,17,18,23,24,31,32,41,42],
    #    "policy_FR" : [0,1,2,3,4,11,12,13,14,15,16,17,18,25,26,33,34,35,36]
    #}
    
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
        self.obs_indices = {}
        self.obs_indices["policy_FL"] = [0,1,2,3,4, 5, 6,13,14,15,16,17,18,19,20,27,28,37,38]
        self.obs_indices["policy_HL"] = [0,1,2,3,4, 7, 8,13,14,15,16,17,18,21,22,29,30,39,40]
        self.obs_indices["policy_HR"] = [0,1,2,3,4, 9,10,13,14,15,16,17,18,23,24,31,32,41,42]
        self.obs_indices["policy_FR"] = [0,1,2,3,4,11,12,13,14,15,16,17,18,25,26,33,34,35,36]
        self.std_scaler = MeanStdFilter((len(self.policy_names), len(self.obs_indices["policy_FL"])))
        self.adj = self.create_adj()
        super().__init__(config)

    def create_edge_index(self):
        policy_idx = list(self.obs_indices.keys())
        get_node_idx = lambda policy_name: policy_idx.index(policy_name)
        make_edge = lambda sender, receiver: [get_node_idx(sender), get_node_idx(receiver)]
        # create bidirectional edge index
        return [make_edge("policy_FL", "policy_HL"),
                make_edge("policy_HL", "policy_HR"),
                make_edge("policy_HR", "policy_FR"),
                make_edge("policy_FR", "policy_FL"),

                make_edge("policy_HL", "policy_FL"),
                make_edge("policy_HR", "policy_HL"),
                make_edge("policy_FR", "policy_HR"),
                make_edge("policy_FL", "policy_FR")]

    def create_adj(self):
        edge_index = self.create_edge_index()
        adj = np.zeros([4, 4], dtype=np.float64)
        adj[(*np.transpose(edge_index),)] = 1.
        return adj

    @staticmethod
    def return_policies(obs_space):
        # For each agent the policy interface has to be defined.
        index_space = spaces.MultiDiscrete([4])
        obs_space = spaces.Box(-np.inf, np.inf, (4, 19,), np.float64)
        adj_space = spaces.MultiDiscrete(np.ones([4, 4]) * 2)
        graph_space = spaces.Tuple([index_space, obs_space, adj_space])
        policies = {
            QuantrupedFullyDecentralizedEnv.policy_names[0]: (None,
                graph_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            QuantrupedFullyDecentralizedEnv.policy_names[1]: (None,
                graph_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            QuantrupedFullyDecentralizedEnv.policy_names[2]: (None,
                graph_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            QuantrupedFullyDecentralizedEnv.policy_names[3]: (None,
                graph_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
        }
        return policies

    def distribute_observations(self, obs_full):
        """ 
        Construct dictionary that routes to each policy only the relevant
        local information.
        """
        obs_distributed = {}
        policy_idx = list(self.obs_indices.keys())
        graph_observation = np.stack([ obs_full[self.obs_indices[p_idx]] for p_idx in policy_idx ])
        graph_observation = self.std_scaler(graph_observation)

        for policy_name in self.policy_names:
            obs_distributed[policy_name] = (
                np.array([policy_idx.index(policy_name)]), 
                graph_observation.astype(obs_full.dtype), 
                self.adj
            )

        return obs_distributed

class QuantrupedFullyDecentralizedEnv(QuantrupedFourControllerSuperEnv):
    """ Derived environment for control of the four-legged agent.
        Uses four different, concurrent control units (policies) 
        each instantiated as a single agent. 
        
        Input scope of each controller: only the controlled leg.
        
        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in the obs_indices for each leg.
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
        self.obs_indices["policy_FL"] = [0,1,2,3,4, 5, 6,13,14,15,16,17,18,19,20,27,28,37,38]
        self.obs_indices["policy_HL"] = [0,1,2,3,4, 7, 8,13,14,15,16,17,18,21,22,29,30,39,40]
        self.obs_indices["policy_HR"] = [0,1,2,3,4, 9,10,13,14,15,16,17,18,23,24,31,32,41,42]
        self.obs_indices["policy_FR"] = [0,1,2,3,4,11,12,13,14,15,16,17,18,25,26,33,34,35,36]
        super().__init__(config)
            
    @staticmethod
    def return_policies(obs_space):
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (19,), np.float64)
        policies = {
            QuantrupedFullyDecentralizedEnv.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            QuantrupedFullyDecentralizedEnv.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            QuantrupedFullyDecentralizedEnv.policy_names[2]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            QuantrupedFullyDecentralizedEnv.policy_names[3]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
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
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in the obs_indices for each leg.
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
        # FL also gets local information from HL
        self.obs_indices["policy_FL"] = [0,1,2,3,4, 5, 6, 7, 8,13,14,15,16,17,18,19,20,21,22,27,28,29,30,37,38,39,40]
        # HL also gets local information from HR
        self.obs_indices["policy_HL"] = [0,1,2,3,4, 7, 8, 9,10,13,14,15,16,17,18,21,22,23,24,29,30,31,32,39,40,41,42]
        # HR also gets local information from FR
        self.obs_indices["policy_HR"] = [0,1,2,3,4, 9,10,11,12,13,14,15,16,17,18,23,24,25,26,31,32,33,34,41,42,35,36]
        # FR also gets local information from FL
        self.obs_indices["policy_FR"] = [0,1,2,3,4,11,12, 5, 6,13,14,15,16,17,18,25,26,19,20,33,34,27,28,35,36,37,38]
        super().__init__(config) 
            
    @staticmethod
    def return_policies(obs_space):
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (27,), np.float64)
        policies = {
            Quantruped_LocalSingleNeighboringLeg_Env.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_LocalSingleNeighboringLeg_Env.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_LocalSingleNeighboringLeg_Env.policy_names[2]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_LocalSingleNeighboringLeg_Env.policy_names[3]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
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
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in the obs_indices for each leg.
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
        self.obs_indices["policy_FL"] = [0,1,2,3,4, 5, 6, 9,10,13,14,15,16,17,18,19,20,23,24,27,28,31,32,37,38,41,42]
        self.obs_indices["policy_HL"] = [0,1,2,3,4, 7, 8,11,12,13,14,15,16,17,18,21,22,25,26,29,30,33,34,39,40,35,36]
        self.obs_indices["policy_HR"] = self.obs_indices["policy_FL"] #[0,1,2,3,4, 9,10,13,14,15,16,17,18,23,24,31,32,41,42]
        self.obs_indices["policy_FR"] = self.obs_indices["policy_HL"] #[0,1,2,3,4,11,12,13,14,15,16,17,18,25,26,33,34,35,36]
        super().__init__(config)
            
    @staticmethod
    def return_policies(obs_space):
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (27,), np.float64)
        policies = {
            Quantruped_LocalSingleDiagonalLeg_Env.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_LocalSingleDiagonalLeg_Env.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_LocalSingleDiagonalLeg_Env.policy_names[2]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_LocalSingleDiagonalLeg_Env.policy_names[3]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
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
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in the obs_indices for each leg.
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
        # FL also gets local information from HL (towards front)
        self.obs_indices["policy_FL"] = [0,1,2,3,4, 5, 6, 7, 8,13,14,15,16,17,18,19,20,21,22,27,28,29,30,37,38,39,40]
        # HL also gets local information from HR (from side at back)
        self.obs_indices["policy_HL"] = [0,1,2,3,4, 7, 8, 9,10,13,14,15,16,17,18,21,22,23,24,29,30,31,32,39,40,41,42]
        # HR also gets local information from HL (from side at back)
        self.obs_indices["policy_HR"] = [0,1,2,3,4, 9,10, 7, 8,13,14,15,16,17,18,23,24,21,22,31,32,29,30,41,42,39,40]
        # FR also gets local information from HR (towards front)
        self.obs_indices["policy_FR"] = [0,1,2,3,4,11,12, 9,10,13,14,15,16,17,18,25,26,23,24,33,34,31,32,35,36,41,42]
        super().__init__(config) 
            
    @staticmethod
    def return_policies(obs_space):
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (27,), np.float64)
        policies = {
            Quantruped_LocalSingleToFront_Env.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_LocalSingleToFront_Env.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_LocalSingleToFront_Env.policy_names[2]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_LocalSingleToFront_Env.policy_names[3]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
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
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in the obs_indices for each leg.
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
        # FL also gets local information from HL and FR
        self.obs_indices["policy_FL"] = [0,1,2,3,4, 5, 6, 7, 8,11,12,13,14,15,16,17,18,19,20,21,22,25,26,27,28,29,30,33,34,37,38,39,40,35,36]
        # HL also gets local information from HR and FL
        self.obs_indices["policy_HL"] = [0,1,2,3,4, 7, 8, 9,10, 5, 6,13,14,15,16,17,18,21,22,23,24,19,20,29,30,31,32,27,28,39,40,41,42,37,38]
        # HR also gets local information from FR and HL
        self.obs_indices["policy_HR"] = [0,1,2,3,4, 9,10,11,12, 7, 8,13,14,15,16,17,18,23,24,25,26,21,22,31,32,33,34,29,30,41,42,35,36,39,40]
        # FR also gets local information from FL and HR
        self.obs_indices["policy_FR"] = [0,1,2,3,4,11,12, 5, 6, 9,10,13,14,15,16,17,18,25,26,19,20,23,24,33,34,27,28,31,32,35,36,37,38,41,42]
        super().__init__(config)
            
    @staticmethod
    def return_policies(obs_space):
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (35,), np.float64)
        policies = {
            Quantruped_Local_Env.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_Local_Env.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_Local_Env.policy_names[2]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            Quantruped_Local_Env.policy_names[3]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
        }
        return policies
