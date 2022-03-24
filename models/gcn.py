import tensorflow as tf
import tensorflow.keras.layers as layers

import models.graph_ops as graph_ops


class GCN(layers.Layer):
    '''
    Implements vanilla graph convolution as presented in this 
    paper: https://arxiv.org/abs/1609.02907
    Effectively, this implements: activation(A' * X  * W),
    where A' is the normalized adjacency matrix, X are the node
    features and W is the weight matrix.
    '''
    def __init__(self, units, activation=None, use_weight=True, use_bias=False, **kwargs):
        super(GCN, self).__init__()
        self.bias = None
        if use_weight:
            self.linear = layers.Dense(units, use_bias=False)
        self.activation = layers.Activation(activation)
        self.use_weight = use_weight
        self.use_bias = use_bias
        self.units = units

    def build(self, input_shape):
        if self.use_bias:
            self.bias = tf.Variable(tf.zeros(self.units), dtype=tf.float32, trainable=True, name='GCN_bias')

    def call(self, x, adj):
        adj_normed = graph_ops.adj_norm(adj)
        x = adj_normed @ x
        x = self.linear(x)

        if self.use_bias:
            x += self.bias

        return self.activation(x)

class MPNN(layers.Layer):

    def __init__(self, units, activation=None, use_weight=True, use_bias=False, **kwargs):
        super(MPNN, self).__init__()
        self.bias = None
        if use_weight:
            self.msg_transform = layers.Dense(units, use_bias=False)
            self.node_update = layers.Dense(units, use_bias=False)
        self.activation = layers.Activation(activation)
        self.use_weight = use_weight
        self.use_bias = use_bias
        self.units = units

    def build(self, input_shape):
        if self.use_bias:
            self.bias = tf.Variable(tf.zeros(self.units), dtype=tf.float32, trainable=True, name='MPNN_bias')

    def call(self, x, adj):
        # adj is batched, so edge_index will be [n_edges, 3]
        # provide indices for batch, node, features
        edge_index = tf.where(adj)
        n_batches, n_nodes = tf.unstack(tf.cast(tf.shape(adj)[:2], tf.int64))
        batch, senders, receivers = tf.unstack(edge_index, axis=-1)

        senders = tf.stack((batch, senders), axis=-1)
        receivers = tf.stack((batch, receivers), axis=-1)

        batch_mod = batch * n_nodes

        x_snd = tf.gather_nd(x, senders)
        #x_rec = tf.gather_nd(x, receivers)

        messages = tf.math.unsorted_segment_mean(
            data=x_snd, 
            segment_ids=receivers[:,1] + batch_mod, 
            num_segments=n_nodes * n_batches)

        messages = self.msg_transform(messages)

        x = self.node_update(x)
        x_shape = tf.shape(x)
        
        # put back into original batches
        # this only works if every batch has a similar number of nodes,
        # in this application this is the case: the batches contain always the same graph
        messages = tf.reshape(messages, x_shape)

        x = x + messages
        
        if self.use_bias:
            x += self.bias


        return self.activation(x)