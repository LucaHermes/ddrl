import numpy as np

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import get_activation_fn, try_import_tf

from models.glorot_uniform_scaled_initializer import GlorotUniformScaled
from models.gcn import GCN
from models.graph_net import GraphNet


tf1, tf, tfv = try_import_tf()


class FullyConnectedNetwork_GNN_GlorotUniformInitializer(TFModelV2):
    """ A fully connected Network - same as the provided generic one
        (ray.rllib.models.tf.FullyConnectedNetwork), but using the
        Glorot Uniform initialization (scaled for action output which
        requires a derived scaled initializer).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(FullyConnectedNetwork_GNN_GlorotUniformInitializer, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)
        activation = get_activation_fn(model_config.get("fcnet_activation"))
        hiddens = model_config.get("fcnet_hiddens", [])
        no_final_linear = model_config.get("no_final_linear")
        vf_share_layers = model_config.get("vf_share_layers")
        free_log_std = model_config.get("free_log_std")

        self_node_id_space, obs_space, adj_space = obs_space

        self.actor = GraphNet(num_outputs, model_config)
        self.critic = GraphNet(1, model_config)

        # We have to build these models to initialize the trainable parameters
        # We build by calling the model (instead of the build method) because tensorflow 
        # only supports layers with float inputs when using build.
        self.actor(
            tf.zeros((1, *self_node_id_space.shape), dtype=tf.int32),
            tf.zeros((1, *obs_space.shape)),
            tf.zeros((1, *adj_space.shape))
        )
        self.critic(
            tf.zeros((1, *self_node_id_space.shape), dtype=tf.int32),
            tf.zeros((1, *obs_space.shape)),
            tf.zeros((1, *adj_space.shape))
        )

        self.register_variables(self.actor.variables)
        self.register_variables(self.critic.variables)

    def forward(self, input_dict, state, seq_lens):
        action = self.actor(*input_dict["obs"])
        self._value_out = self.critic(*input_dict["obs"])
        return action, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class FullyConnectedNetwork_GNN_GlorotUniformInitializer_GlobalCritic(TFModelV2):
    """ A fully connected Network - same as the provided generic one
        (ray.rllib.models.tf.FullyConnectedNetwork), but using the
        Glorot Uniform initialization (scaled for action output which
        requires a derived scaled initializer).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(FullyConnectedNetwork_GNN_GlorotUniformInitializer, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)
        activation = get_activation_fn(model_config.get("fcnet_activation"))
        hiddens = model_config.get("fcnet_hiddens", [])
        no_final_linear = model_config.get("no_final_linear")
        vf_share_layers = model_config.get("vf_share_layers")
        free_log_std = model_config.get("free_log_std")

        self_node_id_space, obs_space, adj_space = obs_space

        self.actor = GraphNet(num_outputs, model_config)
        self.critic = tf.keras.Sequential([
            tf.keras.layers.Dense(hiddens[0], activation='tanh', kernel_initializer=GlorotUniformScaled(1.0)),
            tf.keras.layers.Dense(hiddens[1], activation='tanh', kernel_initializer=GlorotUniformScaled(1.0)),
            tf.keras.layers.Dense(1, kernel_initializer=GlorotUniformScaled(0.01))
        ])
        # We have to build these models to initialize the trainable parameters
        # We build by calling the model (instead of the build method) because tensorflow 
        # only supports layers with float inputs when using build.
        self.actor(
            tf.zeros((1, *self_node_id_space.shape), dtype=tf.int32),
            tf.zeros((1, *obs_space.shape)),
            tf.zeros((1, *adj_space.shape))
        )
        self.critic(
            tf.zeros((1, tf.reduce_prod(obs_space.shape))),
        )

        self.register_variables(self.actor.variables)
        self.register_variables(self.critic.variables)

    def forward(self, input_dict, state, seq_lens):
        node_idx = tf.cast(input_dict["obs"][0], tf.int32)
        action = self.actor(*input_dict["obs"])
        #critic_inputs = tf.gather(input_dict["obs"][1], tf.cast(input_dict["obs"][0], tf.int32), axis=1, batch_dims=1)
        # critic gets global information
        critic_inputs = tf.concat(tf.unstack(tf.roll(input_dict["obs"][1], -node_idx[:,0], axis=tf.ones_like(node_idx[:,0])), axis=1), axis=-1)
        self._value_out = self.critic(critic_inputs)
        return action, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

