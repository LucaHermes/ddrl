import tensorflow as tf

def symm_norm(adj):
    '''
    Symmetric adjacency normalization.
    This is simply A' = D^-0.5 @ A @ D^-0.5; the given matrix
    normalized with the inverse of the degree matrix from both sides.
    '''
    d_norm = tf.reduce_sum(adj, axis=-1)**-0.5
    d_norm = tf.linalg.diag(d_norm)
    return d_norm @ adj @ d_norm

def adj_norm(adj):
    '''
    Non-symmetric adjacency normalization.
    This is simply A' = D^-1 @ A; the given matrix
    normalized with the inverse of the degree matrix.
    '''
    d_norm = tf.reduce_sum(adj, axis=-1)**-1.
    d_norm = tf.linalg.diag(d_norm)
    return d_norm @ adj

def segment_softmax(data, segment_ids, num_segments):
    seg_sums = tf.math.unsorted_segment_sum(tf.exp(data), segment_ids, num_segments)
    seg_sums = tf.gather(seg_sums, segment_ids)
    return tf.exp(data) / seg_sums
