from ray.rllib.utils.framework import get_activation_fn, try_import_tf
from models.glorot_uniform_scaled_initializer import GlorotUniformScaled
from models.gcn import GCN, MPNN, GAT1, MPNN2


tf1, tf, tfv = try_import_tf()

class GraphNet(tf.keras.Model):

    def __init__(self, num_outputs, obs_space, model_config):
        super(GraphNet, self).__init__()
        activation = get_activation_fn(model_config.get("fcnet_activation"))
        hiddens = model_config.get("fcnet_hiddens", [])
        self.state_size = obs_space.shape[-1]
        self.enc = tf.keras.layers.Dense(
            #hiddens[0],
            (self.state_size - 4) * hiddens[0],
            name="state_enc",
            activation=activation,
            kernel_initializer=GlorotUniformScaled(1.0))
        self.gnn = MPNN( #MPNN2( # GAT1( #MPNN( #GCN(
            hiddens[1],
            activation=activation,
            use_bias=False,
            kernel_initializer=GlorotUniformScaled(1.0))
        self.out = tf.keras.layers.Dense(
            num_outputs,
            name="linear_out",
            activation=None,
            kernel_initializer=GlorotUniformScaled(0.01))
        self.leg_enc_dim = hiddens[0]

    def enc_leg_features(self, state):
        w = self.enc(state[...,-4:])
        n_nodes = tf.shape(w)[1]
        w = tf.reshape(w, [-1, n_nodes, self.state_size-4, self.leg_enc_dim])
        leg_feats = tf.expand_dims(state[...,:-4], axis=-2)
        return tf.nn.tanh(tf.squeeze(leg_feats @ w, axis=-2))

    def call(self, node_idx, state, adj):
        x = self.enc_leg_features(state) #self.enc(state)
        x = self.gnn(x, adj)
        n_idx = tf.cast(tf.reshape(node_idx, [-1]), tf.int32)
        x = tf.gather(x, n_idx, axis=1, batch_dims=1)
        #x = self.enc(x)
        return self.out(x)
