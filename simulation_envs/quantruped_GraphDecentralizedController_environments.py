from ray.rllib.utils.filter import MeanStdFilter
import numpy as np
from gym import spaces

from simulation_envs import QuantrupedMultiPoliciesEnv

class QuantrupedDecentralizedGraphSuperEnv(QuantrupedMultiPoliciesEnv):

    # This is ordering of the policies as applied here:
    policy_names = ['policy_FL','policy_HL','policy_HR','policy_FR']
    agent_names = ["agent_FL","agent_HL","agent_HR","agent_FR"]

    def __init__(self, config):
        super().__init__(config)
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
        self.obs_indices = {
            'agent_FL' : self.env.get_obs_indices(['body', 'fl']), #[0,1,2,3,4, 5, 6,13,14,15,16,17,18,19,20,27,28,37,38]
            'agent_HL' : self.env.get_obs_indices(['body', 'hl']), #[0,1,2,3,4, 7, 8,13,14,15,16,17,18,21,22,29,30,39,40]
            'agent_HR' : self.env.get_obs_indices(['body', 'hr']),
            'agent_FR' : self.env.get_obs_indices(['body', 'fr']),
        }
        self.std_scaler = MeanStdFilter((len(self.agent_names), len(self.obs_indices['agent_FL'])))
        self.adj = self.create_adj()


class QuantrupedDecentralizedGraphEnv(QuantrupedDecentralizedGraphSuperEnv):
    ''' Derived environment for control of the four-legged agent.
        Uses four different, concurrent control units (policies) 
        each instantiated as a single agent. 

        Input scope of each controller: only the controlled leg.

        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in the obs_indices for each leg.
    ''' 
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        if agent_id.startswith('agent_FL'):
            return 'policy_FL'
        elif agent_id.startswith('agent_HL'):
            return 'policy_HL'
        elif agent_id.startswith('agent_HR'):
            return 'policy_HR'
        else:
            return 'policy_FR'
        
    def create_edge_index(self):
        agent_idx = self.agent_names
        get_node_idx = lambda agent_name: agent_idx.index(agent_name)
        make_edge = lambda sender, receiver: [get_node_idx(sender), get_node_idx(receiver)]
        # create bidirectional edge index
        return [make_edge('agent_FL', 'agent_HL'),
                make_edge('agent_HL', 'agent_HR'),
                make_edge('agent_HR', 'agent_FR'),
                make_edge('agent_FR', 'agent_FL'),

                make_edge('agent_HL', 'agent_FL'),
                make_edge('agent_HR', 'agent_HL'),
                make_edge('agent_FR', 'agent_HR'),
                make_edge('agent_FL', 'agent_FR')]

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
        agent_idx = list(self.obs_indices.keys())
        graph_observation = np.stack([ obs_full[self.obs_indices[a_idx]] for a_idx in agent_idx ])
        graph_observation = self.std_scaler(graph_observation)

        for agent_name in self.agent_names:
            obs_distributed[agent_name] = (
                np.array([agent_idx.index(agent_name)]), 
                graph_observation.astype(obs_full.dtype), 
                self.adj
            )

        return obs_distributed


class QuantrupedDecentralizedSharedGraphEnv(QuantrupedDecentralizedGraphEnv):
    ''' Derived environment for control of the four-legged agent.
        Uses four different, concurrent control units (policies) 
        each instantiated as a single agent. 

        Input scope of each controller: only the controlled leg.

        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in the obs_indices for each leg.
    '''

    policy_names = ['leg_policy']

    leg_angles = {
        'agent_FL' : 45.,
        'agent_HL' : 135.,
        'agent_HR' : -135.,
        'agent_FR' : -45.,
    }
        
    def leg_encoding(self, angle):
        rad = np.deg2rad(angle)
        return np.stack((np.sin(rad), np.cos(rad)))
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        return 'leg_policy'

    def create_edge_index(self):
        agent_idx = self.agent_names
        get_node_idx = lambda agent_name: agent_idx.index(agent_name)
        make_edge = lambda sender, receiver: [get_node_idx(sender), get_node_idx(receiver)]
        # create bidirectional edge index
        return [
            make_edge('agent_FL', 'agent_HL'),
            make_edge('agent_HL', 'agent_HR'),
            make_edge('agent_HR', 'agent_FR'),
            make_edge('agent_FR', 'agent_FL'),

            make_edge('agent_HL', 'agent_FL'),
            make_edge('agent_HR', 'agent_HL'),
            make_edge('agent_FR', 'agent_HR'),
            make_edge('agent_FL', 'agent_FR'),
            # selfloops
            #*[ [i, i] for i in range(len(self.agent_names)) ]
        ]

    def create_adj(self):
        edge_index = self.create_edge_index()
        adj = np.zeros([4, 4], dtype=np.float64)
        adj[(*np.transpose(edge_index),)] = 1.
        return adj

    @staticmethod
    def return_policies(use_target_velocity=False):
        # For each agent the policy interface has to be defined.
        n_dims = 19 + use_target_velocity + 2
        index_space = spaces.MultiDiscrete([4])
        obs_space = spaces.Box(-np.inf, np.inf, (4, n_dims,), np.float64)
        adj_space = spaces.MultiDiscrete(np.ones([4, 4]) * 2)
        graph_space = spaces.Tuple([index_space, obs_space, adj_space])
        policies = {
            QuantrupedDecentralizedSharedGraphEnv.policy_names[0]: (None,
                graph_space, spaces.Box(-1., +1., (2,)), {}),
        }
        return policies

    def distribute_observations(self, obs_full):
        ''' 
        Construct dictionary that routes to each policy only the relevant
        local information.
        '''
        obs_distributed = {}
        agent_idx = list(self.obs_indices.keys())


        obs_full_normed = self._normalize_observation(obs_full)
        get_leg_features = lambda agent_name: np.concatenate((
            obs_full_normed[self.obs_indices[agent_name]], 
            self.leg_encoding(self.leg_angles[agent_name])
        ))
        
        graph_observation = np.stack([ get_leg_features(a_idx) for a_idx in agent_idx ])

        for agent_name in self.agent_names:
            obs_distributed[agent_name] = (
                np.array([agent_idx.index(agent_name)]), 
                graph_observation.astype(obs_full.dtype), 
                self.adj
            )

        return obs_distributed
