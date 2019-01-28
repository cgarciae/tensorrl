import tensorflow as tf
import numpy as np
from tensorrl import utils
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
import os
import time
import cv2
import matplotlib.pyplot as plt
# import imageio

def update_target_weights_hard(target_variables, model_variables):
    for target, current in zip(target_variables, model_variables):
        target.assign(current)


def update_target_weights_soft(target_variables, model_variables, beta):
    for target, current in zip(target_variables, model_variables):
        target.assign_add(beta * (current - target))

def play_episodes(model_dir, env, model, policy, visualize, episodes = 1):

    total_reward = [0.0] * episodes
    fourcc = cv2.VideoWriter_fourcc(*"MJPG") # Be sure to use lower case
    videos_path = os.path.join(model_dir, "videos")
    

    for ep in range(episodes):
        state = env.reset()
        terminal = False
        writer = None
        episode_path = os.path.join(videos_path, str(int(time.time() * 1000)))
        os.makedirs(episode_path)

        image_index = 0
        while not terminal:
            image_index += 1
            state = np.expand_dims(state, axis = 0)
            q_values = model(state, training = False)
            action = policy.select_action(q_values = q_values[0].numpy())
            state, reward, terminal, _info = env.step(action)

            if visualize:
                image_path = os.path.join(episode_path, f"{image_index}.jpg")
                image = env.render(mode="rgb_array")[..., ::-1]
                cv2.imwrite(image_path, image)
                
                # plt.imshow(image)

                # if writer is None:
                #     shape = image.shape[:2][::-1]
                #     videos_path = os.path.join(model_dir, "videos")
                #     os.makedirs(videos_path, exist_ok=True)
                #     output_path = os.path.join(videos_path, str(int(time.time())) + ".gif")
                    # writer = cv2.VideoWriter(output_path, fourcc, 20.0, shape, False)
                    # writer = FFmpegWriter(output_path)

                # writer.write(image)
                # writer.writeFrame(image)
                # images.append(image)

            total_reward[ep] += reward

        # if visualize:
        #     imageio.mimwrite(output_path, images, duration = 0.00001)
            # writer.release()
            # writer.close()

        
    return np.mean(total_reward)

        

class DQN(object):

    def __init__(self, model_fn, model_dir, params = {}):

        self.model_fn = model_fn
        self.model_dir = model_dir
        self.params = params
    
    def train(
        self, 
        env, 
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
        visualize_eval = False,
        seed = None,
        double_dqn = False,
        env_cycles = 1,
        train_cycles = 1,
        huber_delta = 100.0,
        eval_episode_frequency = None,
        eval_episodes = 1,
        ):

        min_memory = max(warmup_steps, batch_size) if warmup_steps is not None else batch_size

    

        #####################
        # config
        #####################

        if seed is not None:
            tf.random.set_seed(seed)

        step = 0
        
        #####################
        # model_fn
        #####################
        
        model: tf.keras.Model = self.model_fn()
        target_model: tf.keras.Model = self.model_fn()

        
        # get variables
        model_variables = model.variables
        target_variables = target_model.variables


        def loss_fn(rewards, actions, terminal, model_q_values, target_q_values):
        
            not_terminal = 1.0 - tf.cast(terminal, tf.float32)

            if double_dqn:
                model_actions = tf.argmax(model_q_values, axis=1, output_type=tf.int32)
                action_values = utils.select_columns(target_q_values, model_actions)
            else:
                action_values = tf.reduce_max(target_q_values, axis=1)

            target_values = rewards + gamma * action_values * not_terminal
            

            model_action_values = utils.select_columns(model_q_values, actions)

            loss = utils.huber_loss(target_values, model_action_values, delta = huber_delta)

            return loss


        optimizer = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate)


        #####################
        # initializers
        #####################

        state = env.reset()
        
        
        _episode = 0
        _episode_length = 0
        _episode_reward = 0.0

        while optimizer.iterations < max_steps:

            #################
            # env cycles
            #################

            for _ in range(env_cycles):
            
                state = np.expand_dims(state, axis = 0)
                predictions = model(state, training = False)


                action = policy.select_action(q_values = predictions[0].numpy())

                state1, reward, terminal, _info = env.step(action)

                if visualize:
                    env.render()

                #
                _episode_length += 1
                _episode_reward += reward
                #

                memory.append(state, action, reward, terminal)

                if terminal:


                    if eval_episode_frequency and (_episode % eval_episode_frequency) == 0:
                        mean_total_reward = play_episodes(
                            self.model_dir, env, model, GreedyQPolicy(), 
                            visualize_eval, episodes=eval_episodes)
                        
                        print(f"Episode: {_episode}, Mean Total Reward: {mean_total_reward}")
                    
                    state = state1 = env.reset()
                    #
                    _episode_length = 0
                    _episode_reward = 0.0
                    _episode += 1
                    #

                
                state = state1

            #################
            # train cycles
            #################

            if memory.nb_entries > min_memory:
                for _ in range(train_cycles):

                    step += 1
                
                    experiences = memory.sample(batch_size)

                    state_batch, action_batch, reward_batch, state1_batch, terminal_batch = [ np.asarray(x).squeeze() for x in zip(*experiences) ]

                    state_batch = state_batch.astype(np.float32)
                    state1_batch = state1_batch.astype(np.float32)

                    target_q_values = target_model(state1_batch, training = False)

                    with tf.GradientTape() as tape:
                        model_q_values = model(state_batch, training = True)
                        loss = loss_fn(reward_batch, action_batch, terminal_batch, model_q_values, target_q_values)

                    gradients = tape.gradient(loss, model_variables)
                    gradients = zip(gradients, model_variables)

                    optimizer.apply_gradients(gradients)


                    if target_model_update >= 1 and tf.equal(tf.train.get_global_step() % target_model_update,  0):
                        update_target_weights_hard(target_variables, model_variables)
                    else:
                        update_target_weights_soft(target_variables, model_variables, target_model_update)

                
                


            

                

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
                model_q_values = self.model_fn(model_inputs, tf.estimator.ModeKeys.PREDICT, self.params)


            
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

                predictions = sess.run(model_q_values, step_feed)

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
                