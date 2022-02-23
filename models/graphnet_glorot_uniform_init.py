import numpy as np

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import get_activation_fn, try_import_tf

from models.glorot_uniform_scaled_initializer import GlorotUniformScaled
from models.gcn import GCN


tf1, tf, tfv = try_import_tf()

SHARED_GNN = None

class FullyConnectedNetwork_SharedGNN_GlorotUniformInitializer(TFModelV2):
    """ A fully connected Network - same as the provided generic one
        (ray.rllib.models.tf.FullyConnectedNetwork), but using the
        Glorot Uniform initialization (scaled for action output which
        requires a derived scaled initializer).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(FullyConnectedNetwork_SharedGNN_GlorotUniformInitializer, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)
        global SHARED_GNN
        activation = get_activation_fn(model_config.get("fcnet_activation"))
        hiddens = model_config.get("fcnet_hiddens", [])
        no_final_linear = model_config.get("no_final_linear")
        vf_share_layers = model_config.get("vf_share_layers")
        free_log_std = model_config.get("free_log_std")

        self_node_id_space, obs_space, adj_space = obs_space.original_space

        if SHARED_GNN is None:
            in_feats = obs_space.shape[-1]
            out_feats = model_config.get("fcnet_hiddens", [64])[0]
            SHARED_GNN = GCN(out_feats, activation=activation, use_bias=True)

        # Generate free-floating bias variables for the second half of
        # the outputs.
        if free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two", num_outputs)
            num_outputs = num_outputs // 2
            self.log_std_var = tf.Variable(
                [0.0] * num_outputs, dtype=tf.float32, name="log_std")
            self.register_variables([self.log_std_var])

        # Input of the leg id
        node_id_input = tf.keras.layers.Input(
            shape=self_node_id_space.shape,
            dtype=self_node_id_space.dtype,
            name="self_id_input")

        # Input of the leg observation
        inputs = tf.keras.layers.Input(
            shape=obs_space.shape,
            dtype=np.float32, #obs_space.dtype,
            name="observations")

        # Input of the leg observation
        adj_input = tf.keras.layers.Input(
            shape=adj_space.shape,
            dtype=np.float32, #obs_space.dtype,
            name="adj_input")

        # gather the correct observation, i.e. use the leg id
        # to access the corresponding feature vector
        gather = tf.keras.layers.Lambda(
            lambda x: tf.gather(x[0], tf.reshape(x[1], [-1]), 
                axis=1, batch_dims=1))

        # compute message passing on the body graph
        graph_messages = SHARED_GNN(inputs, adj_input)
        # concat the messages to the local input features
        last_layer = tf.keras.layers.Concatenate(axis=-1)([inputs, graph_messages])
        # gather the entry for the leg that is controled
        leg_controller_inputs = gather((inputs, node_id_input))
        last_layer = leg_controller_inputs


        # The action distribution outputs.
        logits_out = None
        i = 1

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_{}".format(i),
                activation=activation,
                kernel_initializer=GlorotUniformScaled(1.0))(last_layer)
            i += 1

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            logits_out = tf.keras.layers.Dense(
                num_outputs,
                name="fc_out",
                activation=activation,
                kernel_initializer=GlorotUniformScaled(1.0))(last_layer)
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                last_layer = tf.keras.layers.Dense(
                    hiddens[-1],
                    name="fc_{}".format(i),
                    activation=activation,
                    kernel_initializer=GlorotUniformScaled(1.0))(last_layer)
            if num_outputs:
                logits_out = tf.keras.layers.Dense(
                    num_outputs,
                    name="fc_out",
                    activation=None,
                    kernel_initializer=GlorotUniformScaled(0.01))(last_layer)
            # Adjust num_outputs to be the number of nodes in the last layer.
            else:
                self.num_outputs = (
                    [int(np.product(obs_space.shape)//obs_space.shape[-1])] + hiddens[-1:])[-1]

        # Concat the log std vars to the end of the state-dependent means.
        if free_log_std and logits_out is not None:

            def tiled_log_std(x):
                return tf.tile(
                    tf.expand_dims(self.log_std_var, 0), [tf.shape(x)[0], 1])

            log_std_out = tf.keras.layers.Lambda(tiled_log_std)(leg_controller_inputs)
            logits_out = tf.keras.layers.Concatenate(axis=1)(
                [logits_out, log_std_out])

        last_vf_layer = None
        if not vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            last_vf_layer = leg_controller_inputs
            i = 1
            for size in hiddens:
                last_vf_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_value_{}".format(i),
                    activation=activation,
                    kernel_initializer=GlorotUniformScaled(1.0))(last_vf_layer)
                i += 1

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=GlorotUniformScaled(0.01))(
                last_vf_layer if last_vf_layer is not None else last_layer)

        self.base_model = tf.keras.Model(
            (node_id_input, inputs, adj_input), [(logits_out
                      if logits_out is not None else last_layer), value_out])

        self.name = name
        self.register_variables(self.base_model.variables)
        self.register_variables(SHARED_GNN.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
