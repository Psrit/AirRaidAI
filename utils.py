import tensorflow as tf


def sync(net_origin: tf.keras.Model, net_target: tf.keras.Model):
    for var_origin, var_target in zip(
        net_origin.trainable_weights, net_target.trainable_weights
    ):
        var_target.assign(var_origin)


def huber_loss(x):
    return tf.where(
        tf.abs(x) < 1, 0.5 * tf.square(x), tf.abs(x) - 0.5
    )
