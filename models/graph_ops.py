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
    print(d_norm @ adj)
    return d_norm @ adj

    x_sum = tf.math.unsorted_segment_sum(x, segment_ids, num_segments=num_segments)
    x_sum = tf.gather(x_sum, segment_ids)
    return x / x_sum