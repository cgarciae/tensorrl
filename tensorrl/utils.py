import tensorflow as tf

def initialize_or_restore(sess, model_dir, global_variables_initializer, saver):

    checkpoint_path = tf.train.latest_checkpoint(model_dir)

    

    if checkpoint_path is None:
        print("Initializing Variables...")
        sess.run(global_variables_initializer)
        
    else:
        print("Restoring Variables from {checkpoint_path}".format(checkpoint_path=checkpoint_path))
        saver.restore(sess, checkpoint_path)


def select_columns(tensor, indices):
    # idx = tf.stack((tf.range(tf.shape(indices)[0]), indices), 1)
    # return tf.gather_nd(tensor, idx)

    # prepare row indices
    row_indices = tf.range(tf.shape(indices)[0])

    # zip row indices with column indices
    full_indices = tf.stack([row_indices, indices], axis=1)

    # retrieve values by indices
    return tf.gather_nd(tensor, full_indices)


def episode_mean(value, done, name = None):

    with tf.variable_scope(name, default_name="EpisodeMean"):

        total = tf.get_variable("total", shape=[], dtype=tf.float32)
        count = tf.get_variable("total", shape=[], dtype=tf.int32)

        ep_mean = total / count

        update = tf.cond(
            done,
            lambda: tf.group(
                count.assign(0),
                total.assign(0.0),
            ),
            lambda: tf.group(
                count.assign_add(1),
                total.assign_add(value),
            ),
        )

        return ep_mean, update

# @tf.function
def huber_loss(labels, predictions, delta=1.0):
    a = predictions - labels

    loss = tf.where(
        tf.abs(a) <= delta,
        0.5 * tf.square(a),
         delta * tf.abs(a) - 0.5 * delta ** 2
    )

    loss = tf.reduce_mean(loss)
    
    return loss