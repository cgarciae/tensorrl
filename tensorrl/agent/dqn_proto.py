import tensorflow as tf
import numpy as np
from tensorrl import utils
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

def update_target_weights(target_variables, model_variables, target_model_update):

    hard_update = target_model_update >= 1

    updates = [ 
        target.assign(current) if hard_update else
        target.assign_add(target_model_update * (current - target))
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
        policy = EpsGreedyQPolicy(),
        memory = SequentialMemory(limit = 1000, window_length=1),
        target_model_update = 10000,
        gamma = 0.99,
        warmup_steps = None,
        batch_size = 64,
        summary_steps = 100,
        visualize = False,
        ):

        min_memory = max(warmup_steps, batch_size) if warmup_steps is not None else batch_size

        with tf.Graph().as_default() as graph:

            inputs = input_fn()

            tf.train.get_or_create_global_step()

            #####################
            # start model_fn
            

            with tf.variable_scope("Model") as model_scope:
                model_inputs = dict(state0 = inputs["state0"])
                model_q_values = self.model_fn(model_inputs, tf.estimator.ModeKeys.TRAIN, self.params)
                model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope.name)

            # with tf.variable_scope("Model", reuse = True) as model_scope:
            #     model_inputs = dict(state0 = inputs["state0"])
            #     predict_q_values = self.model_fn(model_inputs, tf.estimator.ModeKeys.PREDICT, self.params)
            predict_q_values = model_q_values

            with tf.variable_scope("TargetModel") as target_scope:
                target_model_inputs = dict(state0 = inputs["state1"])
                target_q_values = self.model_fn(target_model_inputs, tf.estimator.ModeKeys.PREDICT, self.params)
                target_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_scope.name)

            
            target_values = tf.where(
                inputs["terminal"],
                inputs["reward"],
                inputs["reward"] + gamma * tf.reduce_max(target_q_values, axis=1)
            )
            

            model_action_values = utils.select_columns(model_q_values, inputs["action"])
            

            tf.losses.huber_loss(target_values, model_action_values)
            # loss = tf.reduce_mean(loss)

            loss = tf.losses.get_total_loss()

            optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate)
            
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())


            tf.summary.scalar("target", tf.reduce_mean(target_values))
            
            # end model_fn
            #####################

            #####################
            # train stuff

            tf.summary.scalar("loss", loss)



            train_summaries = tf.summary.merge_all()


            if target_model_update >= 1:
                maybe_update_target = tf.cond(
                    # global_step % target_model_update == 0
                    tf.equal(
                        tf.mod(tf.train.get_global_step(), target_model_update), 
                        0,
                    ),
                    lambda: update_target_weights(target_variables, model_variables, target_model_update),
                    lambda: tf.no_op(),
                )
            else:
                maybe_update_target = update_target_weights(target_variables, model_variables, target_model_update)

            final_train_op = tf.group(
                train_op,
                maybe_update_target,
            )

            
            # train stuff
            #####################

            #####################
            # unautomated episode stuff

            
            
            # unautomated stuff
            #####################

            #####################
            # episode stuff

            

            episode_length_t = tf.placeholder(tf.int32, name="episode_length")
            episode_reward_t = tf.placeholder(tf.int32, name="episode_reward")

            episode_length_summary = tf.summary.scalar("episode_length", episode_length_t)
            episode_reward_summary = tf.summary.scalar("episode_reward", episode_reward_t)
            
            
            final_episode_op = tf.group()
            episode_summaries = tf.summary.merge([
                episode_length_summary,
                episode_reward_summary
            ])


            # episode stuff
            #####################

            state0_t, reward_t, terminal_t, action_t, state1_t = [ inputs[x] for x in ["state0", "reward", "terminal", "action", "state1"] ]

            global_variables_initializer = tf.global_variables_initializer()


        graph.finalize()

        writer = tf.summary.FileWriter(self.model_dir)

        with tf.Session(graph = graph) as sess:
            
            utils.initialize_or_restore(sess, self.model_dir, global_variables_initializer)

            current_step = sess.run(tf.train.get_global_step())

            state0 = env.reset()
            
            _episode_length = 0
            _episode_reward = 0.0

            for step in range(current_step, max_steps):
                
                step_feed = {
                    state0_t : [state0]
                }

                predictions, _ = sess.run([predict_q_values, maybe_update_target], step_feed)

                action = policy.select_action(predictions[0])

                state1, reward, terminal, _info = env.step(action)

                if visualize:
                    env.render()

                #
                _episode_length += 1
                _episode_reward += reward
                #

                memory.append(state0, action, reward, terminal)

                train_fetches = {}
                train_feed = {}

                if memory.nb_entries >= min_memory:
                    experiences = memory.sample(batch_size)
                    experiences = [ list(x) for x in zip(*experiences) ]

                    state0_a, action_a, reward_a, state1_a, terminal_a = experiences

                    state0_a = np.squeeze(state0_a)
                    state1_a = np.squeeze(state1_a)

                    train_feed.update({
                        state0_t : state0_a,
                        action_t : action_a,
                        reward_t : reward_a,
                        state1_t : state1_a,
                        terminal_t : terminal_a,
                    })

                    train_fetches["train_op"] = final_train_op

                    if step % summary_steps == 0:
                        train_fetches["train_summaries"] = train_summaries

                if terminal:
                    train_feed[episode_length_t] = _episode_length
                    train_feed[episode_reward_t] = _episode_reward

                    train_fetches["episode_op"] = final_episode_op
                    train_fetches["episode_summaries"] = episode_summaries

                if step % summary_steps == 0:
                    pass
                
                

                # do training
                results = sess.run(train_fetches, train_feed)

                if "train_summaries" in results:
                    writer.add_summary(
                        results["train_summaries"],
                        step,
                    )

                if "episode_summaries" in results:
                    writer.add_summary(
                        results["episode_summaries"],
                        step,
                    )


                # end step
                if terminal:
                    state0 = env.reset()
                    #
                    _episode_length = 0
                    _episode_reward = 0.0
                    #

                else:
                    state0 = state1