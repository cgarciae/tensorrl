import tensorflow as tf
import rl as krl
from tensorrl import utils

def update_target_weights(target_variables, model_variables):

    updates = [ 
        target.assign(current) 
        for target, current in zip(target_variables, model_variables) 
    ]

    return tf.group(*updates)

class DQN(object):

    def __init__(self, model_fn, model_dir, params = {}):

        self.model_fn = model_fn
        self.model_dir = model_dir
        self.params = params
    
    def train(
        self, env, input_fn, 
        max_steps = 10000, 
        policy = krl.policy.EpsGreedyQPolicy(), 
        memory = krl.memory.SequentialMemory(), 
        target_model_update = 10000,
        gamma = 0.99):

        

        with tf.Graph().as_default() as graph:

            inputs = input_fn()

            keys = ["state0", "reward", "done", "action", "state1"]
            state0, reward, done, action, state1 = [ inputs[x] for x in keys ]

            global_step = tf.train.get_or_create_global_step()
            update_global_step = global_step.assign_add(1)

            #####################
            # start model_fn
            

            with tf.variable_scope("Model") as model_scope:
                model_q_values = self.model_fn(inputs, tf.estimator.ModeKeys.TRAIN, self.params)
                model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope)

            with tf.variable_scope("TargetModel") as target_scope:
                target_q_values = self.model_fn(inputs, tf.estimator.ModeKeys.PREDICT, self.params)
                target_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_scope)

            
            target_values = tf.where(
                done,
                reward,
                reward + gamma * tf.reduce_max(target_q_values, axis=1)
            )

            model_action_values = utils.select_columns(model_q_values, actions)
            error = target_values - model_action_values

            tf.losses.huber_loss(target_values, model_action_values)

            loss = tf.losses.get_total_loss()

            optimizer = tf.train.AdamOptimizer()
            
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss)

            

            
            
            # end model_fn
            #####################


            maybe_update_target = tf.cond(
                # global_step % target_model_update == 0
                tf.equals(
                    tf.mod(global_step, target_model_update), 
                    0,
                ),
                lambda: update_target_weights(target_variables, model_variables),
                lambda: tf.no_op(),
            )

            




        graph.finalize()

        with tf.Session(graph = graph) as sess:
            
            utils.initialize_or_restore(sess, self.model_dir)

            current_step = sess.run(global_step)

            state0 = env.reset()

            for step in range(current_step, max_steps):

                


            

