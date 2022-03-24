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
    def __init__(self, units, activation=None, use_weight=True, use_bias=False, kernel_initializer=None, **kwargs):
        super(GCN, self).__init__()
        self.bias = None
        if use_weight:
            self.linear = layers.Dense(units, use_bias=False, kernel_initializer=kernel_initializer)
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
