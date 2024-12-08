

def periodic_tf(length_scale, period):
    """Periodic kernel TensorFlow operation."""
    amplitude_tf = tf.constant(1, dtype=tf.float64)
    length_scale_tf = tf.constant(length_scale, dtype=tf.float64)
    period_tf = tf.constant(period, dtype=tf.float64)
    kernel = tfk.ExpSinSquared(
        amplitude=amplitude_tf, 
        length_scale=length_scale_tf,
        period=period_tf)
    return kernel

def periodic(xa, xb, length_scale, period):
    """Evaluate periodic kernel."""
    kernel = periodic_tf(length_scale, period)
    kernel_matrix = kernel.matrix(xa, xb)
    with tf.Session() as sess:
        return sess.run(kernel_matrix)
def get_local_periodic_kernel(periodic_length_scale, period, amplitude, local_length_scale):
    periodic = tfk.ExpSinSquared(amplitude=amplitude, length_scale=periodic_length_scale, period=period)
    local = tfk.ExponentiatedQuadratic(length_scale=local_length_scale)
    return periodic * local