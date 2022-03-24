from ray.rllib.utils.framework import get_activation_fn, try_import_tf
from models.glorot_uniform_scaled_initializer import GlorotUniformScaled
from models.gcn import GCN


tf1, tf, tfv = try_import_tf()

class GraphNet(tf.keras.Model):

    def __init__(self, num_outputs, model_config):
        super(GraphNet, self).__init__()
        activation = get_activation_fn(model_config.get("fcnet_activation"))
        hiddens = model_config.get("fcnet_hiddens", [])
        self.enc = tf.keras.layers.Dense(
            hiddens[0],
            name="state_enc",
            activation=activation,
            kernel_initializer=GlorotUniformScaled(1.0))
        self.gnn = GCN(
            hiddens[1],
            activation=activation,
            use_bias=True,
            kernel_initializer=GlorotUniformScaled(1.0))
        self.out = tf.keras.layers.Dense(
            num_outputs,
            name="linear_out",
            activation=None,
            kernel_initializer=GlorotUniformScaled(0.01))

    def call(self, node_idx, state, adj):
        x = self.enc(state)
        x = self.gnn(x, adj)
        n_idx = tf.cast(tf.reshape(node_idx, [-1]), tf.int32)
        leg_x = tf.gather(x, n_idx, axis=1, batch_dims=1)
        return self.out(leg_x)
