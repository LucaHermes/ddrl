from ray.rllib.utils.filter import MeanStdFilter
import numpy as np
from gym import spaces

from simulation_envs import QuantrupedMultiPoliciesEnv

class QuantrupedDecentralizedGraphEnv(QuantrupedMultiPoliciesEnv):
    ''' Derived environment for control of the four-legged agent.
        Uses four different, concurrent control units (policies) 
        each instantiated as a single agent. 

        Input scope of each controller: only the controlled leg.

        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in the obs_indices for each leg.
    ''' 
    
    # This is ordering of the policies as applied here:
    policy_names = ['policy_FL','policy_HL','policy_HR','policy_FR']

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
        self.obs_indices = {
            'policy_FL' : self.env.get_obs_indices(['body', 'fl']), #[0,1,2,3,4, 5, 6,13,14,15,16,17,18,19,20,27,28,37,38]
            'policy_HL' : self.env.get_obs_indices(['body', 'hl']), #[0,1,2,3,4, 7, 8,13,14,15,16,17,18,21,22,29,30,39,40]
            'policy_HR' : self.env.get_obs_indices(['body', 'hr']),
            'policy_FR' : self.env.get_obs_indices(['body', 'fr']),
        }
        self.std_scaler = MeanStdFilter((len(self.policy_names), len(self.obs_indices['policy_FL'])))
        self.adj = self.create_adj()

        
    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        if agent_id.startswith('policy_FL'):
            return 'policy_FL'
        elif agent_id.startswith('policy_HL'):
            return 'policy_HL'
        elif agent_id.startswith('policy_HR'):
            return 'policy_HR'
        else:
            return 'policy_FR'
        
    def create_edge_index(self):
        policy_idx = list(self.obs_indices.keys())
        get_node_idx = lambda policy_name: policy_idx.index(policy_name)
        make_edge = lambda sender, receiver: [get_node_idx(sender), get_node_idx(receiver)]
        # create bidirectional edge index
        return [make_edge('policy_FL', 'policy_HL'),
                make_edge('policy_HL', 'policy_HR'),
                make_edge('policy_HR', 'policy_FR'),
                make_edge('policy_FR', 'policy_FL'),

                make_edge('policy_HL', 'policy_FL'),
                make_edge('policy_HR', 'policy_HL'),
                make_edge('policy_FR', 'policy_HR'),
                make_edge('policy_FL', 'policy_FR')]

    def create_adj(self):
        edge_index = self.create_edge_index()
        adj = np.zeros([4, 4], dtype=np.float64)
        adj[(*np.transpose(edge_index),)] = 1.
        return adj

    @staticmethod
    def return_policies(use_target_velocity=False):
        # For each agent the policy interface has to be defined.
        n_dims = 19 + use_target_velocity
        index_space = spaces.MultiDiscrete([4])
        obs_space = spaces.Box(-np.inf, np.inf, (4, n_dims,), np.float64)
        adj_space = spaces.MultiDiscrete(np.ones([4, 4]) * 2)
        graph_space = spaces.Tuple([index_space, obs_space, adj_space])
        policies = {
            QuantrupedDecentralizedGraphEnv.policy_names[0]: (None,
                graph_space, spaces.Box(-1., +1., (2,)), {}),
            QuantrupedDecentralizedGraphEnv.policy_names[1]: (None,
                graph_space, spaces.Box(-1., +1., (2,)), {}),
            QuantrupedDecentralizedGraphEnv.policy_names[2]: (None,
                graph_space, spaces.Box(-1., +1., (2,)), {}),
            QuantrupedDecentralizedGraphEnv.policy_names[3]: (None,
                graph_space, spaces.Box(-1., +1., (2,)), {}),
        }
        return policies

    def distribute_observations(self, obs_full):
        ''' 
        Construct dictionary that routes to each policy only the relevant
        local information.
        '''
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