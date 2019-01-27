import tensorflow as tf
import numpy as np
from tensorrl import utils
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import os

def update_target_weights_hard(target_variables, model_variables):

    updates = [ 
        target.assign(current)
        for target, current in zip(target_variables, model_variables) 
    ]

    return tf.group(*updates)

def update_target_weights_soft(target_variables, model_variables, beta):

    updates = [ 
        target.assign_add(beta * (current - target))
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
        save_steps = 10000,
        visualize = False,
        seed = None,
        double_dqn = False,
        env_cycles = 1,
        train_cycles = 1,
        huber_delta = 100.0,
        ):

        min_memory = max(warmup_steps, batch_size) if warmup_steps is not None else batch_size

        with tf.Graph().as_default() as graph:

            #####################
            # config
            #####################

            if seed is not None:
                tf.set_random_seed(seed)

            global_step = tf.train.get_or_create_global_step()

            


            #####################
            # inputs
            #####################

            inputs = input_fn()
            state0_t, reward_t, terminal_t, action_t, state1_t = [ inputs[x] for x in ["state0", "reward", "terminal", "action", "state1"] ]

            print(inputs)

            #####################
            # model_fn
            #####################
            

            with tf.variable_scope("Model") as model_scope:
                model_inputs = dict(state = inputs["state0"])
                model_q_values = self.model_fn(model_inputs, tf.estimator.ModeKeys.TRAIN, self.params)
            

            with tf.variable_scope("Model", reuse = True) as predict_scope:
                model_inputs = dict(state = inputs["state0"])
                predict_q_values = self.model_fn(model_inputs, tf.estimator.ModeKeys.PREDICT, self.params)

            with tf.variable_scope("TargetModel") as target_scope:
                target_model_inputs = dict(state = inputs["state1"])
                target_q_values = self.model_fn(target_model_inputs, tf.estimator.ModeKeys.PREDICT, self.params)

            
            # get variables
            model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope.name)
            target_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_scope.name)

            
            not_terminal = 1.0 - tf.cast(inputs["terminal"], tf.float32)

            if double_dqn:
                model_actions = tf.argmax(predict_q_values, axis=1, output_type=tf.int32)
                action_values = utils.select_columns(target_q_values, model_actions)
            else:
                action_values = tf.reduce_max(target_q_values, axis=1)

            target_values = inputs["reward"] + gamma * action_values * not_terminal

            
            assert action_values.get_shape().as_list() == inputs["reward"].get_shape().as_list()
            assert action_values.get_shape().as_list() == not_terminal.get_shape().as_list()
            

            model_action_values = utils.select_columns(model_q_values, inputs["action"])
            

            tf.losses.huber_loss(target_values, model_action_values, delta = huber_delta)
            # loss = tf.reduce_mean(loss)

            loss = tf.losses.get_total_loss()

            optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate)
            
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(
                    loss,
                    global_step = tf.train.get_global_step(), 
                    var_list = tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES, 
                        scope=model_scope.name
                    ),
                )


            tf.summary.scalar("target", tf.reduce_mean(target_values))
            

            #####################
            # train stuff
            #####################

            tf.summary.scalar("loss", loss)


            train_summaries = tf.summary.merge_all()


            if target_model_update >= 1:
                update_target_op = tf.cond(
                    # global_step % target_model_update == 0
                    tf.equal(
                        tf.mod(tf.train.get_global_step(), target_model_update), 
                        0,
                    ),
                    lambda: update_target_weights_hard(target_variables, model_variables),
                    lambda: tf.no_op(),
                )
            else:
                update_target_op = update_target_weights_soft(target_variables, model_variables, target_model_update)

            final_train_op = tf.group(
                train_op,
                update_target_op,
            )

            #####################
            # episode stuff
            #####################

            

            episode_length_t = tf.placeholder(tf.int32, name="episode_length")
            episode_reward_t = tf.placeholder(tf.int32, name="episode_reward")

            episode_length_summary = tf.summary.scalar("episode_length", episode_length_t)
            episode_reward_summary = tf.summary.scalar("episode_reward", episode_reward_t)
            
            
            final_episode_op = tf.group()
            episode_summaries = tf.summary.merge([
                episode_length_summary,
                episode_reward_summary
            ])


            #####################
            # initializers
            #####################

            global_variables_initializer = tf.global_variables_initializer()

            saver = tf.train.Saver()
            writer = tf.summary.FileWriter(self.model_dir)
            writer.add_graph(graph)

        with graph.as_default(), tf.Session(graph = graph) as sess:
            
            utils.initialize_or_restore(sess, self.model_dir, global_variables_initializer, saver)
            graph.finalize()
            

            step = sess.run(global_step)

            state0 = env.reset()
            
            _episode_length = 0
            _episode_reward = 0.0

            while step < max_steps:

                #################
                # env cycles
                #################

                for _ in range(env_cycles):
                
                    step_feed = {
                        state0_t : [state0]
                    }

                    predictions = sess.run(predict_q_values, step_feed)

                    action = policy.select_action(q_values = predictions[0])

                    state1, reward, terminal, _info = env.step(action)

                    if visualize:
                        env.render()

                    #
                    _episode_length += 1
                    _episode_reward += reward
                    #

                    memory.append(state0, action, reward, terminal)

                    if terminal:
                        ep_feed = {
                            episode_length_t: _episode_length,
                            episode_reward_t: _episode_reward,
                        }
                        ep_fetches = dict(
                            episode_op = final_episode_op,
                            episode_summaries = episode_summaries,
                        )

                        results = sess.run(ep_fetches, ep_feed)

                        writer.add_summary(
                            results["episode_summaries"],
                            step,
                        )


                        state0 = env.reset()
                        #
                        _episode_length = 0
                        _episode_reward = 0.0
                        #

                    else:
                        state0 = state1

                #################
                # train cycles
                #################

                for _ in range(train_cycles):
                    
                    if memory.nb_entries > min_memory:
                        step += 1

                        experiences = memory.sample(batch_size)
                        experiences = [ list(x) for x in zip(*experiences) ]

                        state0_a, action_a, reward_a, state1_a, terminal_a = experiences

                        state0_a = np.squeeze(state0_a)
                        state1_a = np.squeeze(state1_a)

                        train_feed = {
                            state0_t : state0_a,
                            action_t : action_a,
                            reward_t : reward_a,
                            state1_t : state1_a,
                            terminal_t : terminal_a,
                        }

                        train_fetches = dict(
                            train_op = final_train_op,
                        )

                        if step % summary_steps == 0:
                            train_fetches["train_summaries"] = train_summaries

                        if step % save_steps == 0:
                            checkpoint_path = os.path.join(self.model_dir, "model.ckpt")
                            saver.save(sess, checkpoint_path, global_step=step)
                

                        # do training
                        results = sess.run(train_fetches, train_feed)

                        if "train_summaries" in results:
                            writer.add_summary(
                                results["train_summaries"],
                                step,
                            )
            

                

    def eval(
        self, env, input_fn, 
        max_steps = 10000, 
        policy = EpsGreedyQPolicy(),
        visualize = True,
        seed = None,
        ):

        with tf.Graph().as_default() as graph:

            #####################
            # config
            #####################

            if seed is not None:
                tf.set_random_seed(seed)

            #####################
            # inputs
            #####################

            inputs = input_fn()
            state0_t, reward_t, terminal_t, action_t, state1_t = [ inputs[x] for x in ["state0", "reward", "terminal", "action", "state1"] ]

            print(inputs)

            #####################
            # model_fn
            #####################
            

            with tf.variable_scope("Model") as predict_scope:
                model_inputs = dict(state = inputs["state0"])
                predict_q_values = self.model_fn(model_inputs, tf.estimator.ModeKeys.PREDICT, self.params)


            
            #####################
            # episode stuff
            #####################

            

            episode_length_t = tf.placeholder(tf.int32, name="episode_length")
            episode_reward_t = tf.placeholder(tf.int32, name="episode_reward")

            episode_length_summary = tf.summary.scalar("episode_length", episode_length_t)
            episode_reward_summary = tf.summary.scalar("episode_reward", episode_reward_t)
            
            
            episode_summaries = tf.summary.merge([
                episode_length_summary,
                episode_reward_summary
            ])


            #####################
            # initializers
            #####################

            global_variables_initializer = tf.global_variables_initializer()

            saver = tf.train.Saver()

            writer = tf.summary.FileWriter(
                os.path.join(self.model_dir, "eval")
            )

        with graph.as_default(), tf.Session(graph = graph) as sess:
            
            utils.initialize_or_restore(sess, self.model_dir, global_variables_initializer, saver)

            graph.finalize()

            state0 = env.reset()
            
            _episode_length = 0
            _episode_reward = 0.0

            for step in range(max_steps):
                
                step_feed = {
                    state0_t : [state0]
                }

                predictions = sess.run(predict_q_values, step_feed)

                action = policy.select_action(q_values = predictions[0])

                state1, reward, terminal, _info = env.step(action)

                if visualize:
                    env.render()

                #
                _episode_length += 1
                _episode_reward += reward
                #

                if terminal:
                    ep_feed = {
                        episode_length_t: _episode_length,
                        episode_reward_t: _episode_reward,
                    }
                    ep_fetches = dict(
                        episode_summaries = episode_summaries,
                    )

                    results = sess.run(ep_fetches, ep_feed)

                    writer.add_summary(
                        results["episode_summaries"],
                        step,
                    )


                    state0 = env.reset()
                    #
                    _episode_length = 0
                    _episode_reward = 0.0
                    #

                else:
                    state0 = state1
                