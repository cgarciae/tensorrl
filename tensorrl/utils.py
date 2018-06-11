import tensorflow as tf

def initialize_or_restore(sess, model_dir):

    checkpoint_path = tf.train.latest_checkpoint(model_dir)

    if checkpoint_path is None:
        sess.run(tf.global_variables_initializer())
        
    else:
        meta_path = checkpoint_path + ".meta"

        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, checkpoint_path)


def select_columns(tensor, indexes):
    idx = tf.stack((tf.range(tf.shape(indexes)[0]), indexes), 1)
    return tf.gather_nd(tensor, idx)


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
