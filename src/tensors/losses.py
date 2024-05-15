import tensorflow as tf

def mean_regret(y_true,y_pred):
    return tf.reduce_mean(tf.nn.relu(-tf.cast(y_true >= 1,tf.float32) * y_pred) + tf.nn.relu(tf.cast(y_true < 1,tf.float32) * y_pred))