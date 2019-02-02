import tensorflow as tf
import numpy as np
from tensorrl import utils
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from tensorrl.memory import ReplayMemory
import os
import time
import cv2
from tensorflow.python.ops import summary_ops_v2
from . import utils as eager_utils


def play_episodes(model_dir, env, model, policy, visualize, episodes = 1):

    total_reward = [0.0] * episodes
    videos_path = os.path.join(model_dir, "videos")

    for ep in range(episodes):
        state = env.reset()
        terminal = False
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
                image_path = os.path.join(episode_path, f"{int(time.time() * 10000)}.jpg")
                image = env.render(mode="rgb_array")[..., ::-1]
                cv2.imwrite(image_path, image)
                

            total_reward[ep] += reward
        
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
        memory = ReplayMemory(max_size = 1000),
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
        checkpoint_steps = 100,
        ):

        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            env.seed(seed)

        
        model: tf.keras.Model = self.model_fn()
        target_model: tf.keras.Model = self.model_fn()

        # maybe load_weights
        if os.path.exists(self.model_dir):
            model.load_weights(
                filepath = os.path.join(self.model_dir, "model"),
            )
            target_model.load_weights(
                filepath = os.path.join(self.model_dir, "target_model"),
            )
        model_variables = model.variables
        target_variables = target_model.variables

        summary_writer = summary_ops_v2.create_file_writer(self.model_dir, flush_millis=10000)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate)
        min_memory = max(warmup_steps, batch_size) if warmup_steps is not None else batch_size

        def loss_fn(rewards, actions, terminal, model_q_values, target_q_values):
        
            not_terminal = 1.0 - tf.cast(terminal, tf.float32)

            if double_dqn:
                model_actions = tf.argmax(model_q_values, axis=1, output_type=tf.int32)
                target_action_values = utils.select_columns(target_q_values, model_actions)
            else:
                target_action_values = tf.reduce_max(target_q_values, axis=1)

            target_values = rewards + gamma * target_action_values * not_terminal
            model_action_values = utils.select_columns(model_q_values, actions)
            loss = utils.huber_loss(target_values, model_action_values, delta = huber_delta)

            return loss
        
        episode = 0
        episode_length = 0
        episode_reward = 0.0
        state = env.reset()

        with summary_writer.as_default():
            while optimizer.iterations < max_steps:
                for _ in range(env_cycles):
                    state = np.expand_dims(state, axis = 0)
                    predictions = model(state, training = False)


                    action = policy.select_action(q_values = predictions[0].numpy())
                    state1, reward, terminal, _info = env.step(action)
                    memory.append(
                        state = state, 
                        action = action, 
                        reward = reward,
                        state1 = state1, 
                        terminal = terminal,
                    )
                    episode_length += 1
                    episode_reward += reward

                    if visualize:
                        env.render()

                    if terminal:
                        if eval_episode_frequency and (episode % eval_episode_frequency) == 0:
                            mean_total_reward = play_episodes(
                                self.model_dir, env, model, GreedyQPolicy(), 
                                visualize_eval, episodes=eval_episodes)

                            summary_ops_v2.scalar("mean_total_reward", mean_total_reward, step = optimizer.iterations)
                            print(f"Episode: {episode}, Mean Total Reward: {mean_total_reward}")
                        
                        state = state1 = env.reset()
                        episode_length = 0
                        episode_reward = 0.0
                        episode += 1
                    
                    state = state1

                #################
                # train cycles
                #################

                if memory.size > min_memory:
                    for _ in range(train_cycles):

                        batch = memory.sample(batch_size)

                        target_q_values = target_model(batch["state1"], training = False)

                        with tf.GradientTape() as tape:
                            model_q_values = model(batch["state"], training = True)
                            loss = loss_fn(batch["reward"], batch["action"], batch["terminal"], model_q_values, target_q_values)

                        gradients = tape.gradient(loss, model_variables)
                        gradients = zip(gradients, model_variables)

                        optimizer.apply_gradients(gradients)

                        # checkpoints
                        if tf.equal(optimizer.iterations % checkpoint_steps, 0):
                            model.save_weights(
                                filepath = os.path.join(self.model_dir, "model"),
                                save_format = "tf",
                            )
                            target_model.save_weights(
                                filepath = os.path.join(self.model_dir, "target_model"),
                                save_format = "tf",
                            )

                        # summaries
                        summary_ops_v2.scalar(
                            "mean_target",
                            tf.reduce_mean(target_q_values),
                            step = optimizer.iterations,
                        )

                        # update weights
                        if target_model_update >= 1 and tf.equal(optimizer.iterations % target_model_update,  0):
                            eager_utils.update_target_weights_hard(target_variables, model_variables)
                        else:
                            eager_utils.update_target_weights_soft(target_variables, model_variables, target_model_update)
                