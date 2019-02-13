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


def play_episodes(model_dir, env, model, policy, visualize, create_video=False, episodes=1):

    total_reward = [0.0] * episodes
    videos_path = os.path.join(model_dir, "videos")

    for ep in range(episodes):
        state = env.reset()
        terminal = False

        if create_video:
            episode_path = os.path.join(videos_path, str(int(time.time() * 1000)))
            os.makedirs(episode_path)

        image_index = 0
        while not terminal:
            image_index += 1
            state = np.expand_dims(state, axis=0)
            model_probs, _ = model(state, training=False)
            action = policy.select_action(model_probs[0].numpy())
            state, reward, terminal, _info = env.step(action)

            if visualize:
                if create_video:
                    image_path = os.path.join(episode_path, f"{int(time.time() * 10000)}.jpg")
                    image = env.render(mode="rgb_array")[..., ::-1]
                    cv2.imwrite(image_path, image)
                else:
                    env.render()
                    time.sleep(0.001)

            total_reward[ep] += reward

    return np.mean(total_reward)


class PPO(object):
    def __init__(self, model_fn, model_dir):
        self.model_fn = model_fn
        self.model_dir = model_dir

    def train(
            self,
            env,
            max_steps=10000,
            policy=EpsGreedyQPolicy(),
            memory=ReplayMemory(max_size=1000),
            target_model_update=10000,
            gamma=0.99,
            lambda_=1.00,
            batch_size=64,
            horizon=2048,
            summary_steps=100,
            save_steps=10000,
            visualize=False,
            visualize_eval=False,
            seed=None,
            double_dqn=False,
            env_cycles=1,
            train_cycles=1,
            huber_delta=100.0,
            eval_episode_frequency=None,
            eval_episodes=1,
            checkpoint_steps=100,
            epsilon=0.2,
            initial_epsilon=1.0,
            learning_rate=0.001,
            critic_loss_type="mse",
            beta=0.0,
            actor="model",
    ):
        target_epsilon = epsilon

        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            env.seed(seed)

        model = self.model_fn()
        target_model = self.model_fn()

        # maybe load_weights
        if os.path.exists(self.model_dir):
            model.load_weights(filepath=os.path.join(self.model_dir, "model"), )

        model_variables = model.variables
        target_variables = target_model.variables

        summary_writer = summary_ops_v2.create_file_writer(self.model_dir, flush_millis=10000)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # metrics
        # learning rate
        # _epsilon_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_epsilon - epsilon,
        #     decay_steps=max_steps,
        #     decay_rate=0.96,
        #     staircase=False,
        # )
        # epsilon_schedule = lambda step: initial_epsilon if step < 500 else target_epsilon
        epsilon_schedule = lambda step: target_epsilon

        def loss_fn(
                # arrays
                state,
                action,
                reward,
                state1,
                terminal,
                target_value,
                target_probs,
                advantage,
                returns,
                # tensors
                model_value,
                model_probs,
        ):
            epsilon = epsilon_schedule(optimizer.iterations)

            # actor loss
            model_action_prob = utils.select_columns(model_probs, action)
            target_action_prob = utils.select_columns(target_probs, action)
            ratio = model_action_prob / (target_action_prob + 1E-6)

            actor_loss = tf.minimum(
                advantage * ratio,
                advantage * tf.clip_by_value(
                    ratio,
                    1.0 - epsilon,
                    1.0 + epsilon,
                ),
            )

            # critic loss
            if critic_loss_type == "clipped":
                model_value_clipped = target_value + tf.clip_by_value(
                    model_value - target_value,
                    -epsilon,
                    epsilon,
                )
                critic_loss = tf.maximum(
                    tf.square(model_value - returns),
                    tf.square(model_value_clipped - returns),
                )
            elif critic_loss_type == "mse":
                critic_loss = tf.square(model_value - returns)

            elif critic_loss_type == "huber":
                critic_loss = utils.huber_loss(model_value, returns, delta=huber_delta)

            return -tf.reduce_mean(actor_loss) + tf.reduce_mean(critic_loss)

        episode = 0
        episode_length = 0
        episode_reward = 0.0
        state = env.reset()

        while optimizer.iterations < max_steps:
            for _ in range(env_cycles):
                state = np.expand_dims(state, axis=0)

                if actor == "model":
                    target_probs, target_value = model(state, training=False)
                elif actor == "target":
                    target_probs, target_value = target_model(state, training=False)

                action = policy.select_action(target_probs[0].numpy())
                state1, reward, terminal, _info = env.step(action)

                memory.append(
                    state=state,
                    action=action,
                    reward=reward,
                    state1=state1,
                    terminal=terminal,
                    target_value=target_value,
                    target_probs=target_probs,
                )
                episode_length += 1
                episode_reward += reward

                if visualize:
                    env.render()

                if terminal:
                    if eval_episode_frequency and (episode % eval_episode_frequency) == 0:
                        # mean_total_reward = play_episodes(self.model_dir, env, model, GreedyQPolicy(), visualize_eval, episodes=eval_episodes)
                        mean_total_reward = play_episodes(self.model_dir, env, model, policy, visualize_eval, episodes=eval_episodes)

                        summary_ops_v2.scalar("mean_total_reward", mean_total_reward, step=optimizer.iterations)
                        print(f"Episode: {episode}, Mean Total returnseward: {mean_total_reward}")

                    state = state1 = env.reset()
                    episode_length = 0
                    episode_reward = 0.0
                    episode += 1

                state = state1

            if memory.size == horizon:

                ################################
                # calculate advantage + returns
                #################################

                not_terminal = 1.0 - memory["terminal"]
                advantage = np.zeros_like(memory["reward"])

                if lambda_ < 1:
                    delta = memory["reward"][:-1] + gamma * memory["target_value"][1:] * not_terminal[:-1] - memory["target_value"][:-1]

                    for i in reversed(range(horizon - 1)):
                        advantage[i] = delta[i] + advantage[i + 1] * (gamma * lambda_) * not_terminal[i]

                else:
                    advantage[-1] = memory["target_value"][-1]

                    for i in reversed(range(horizon - 1)):
                        advantage[i] = memory["reward"][i] + advantage[i + 1] * gamma * not_terminal[i]

                    advantage = advantage - memory["target_value"]

                memory["returns"] = advantage + memory["target_value"]
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1E-6)
                advantage[-1] = 0.0
                memory["advantage"] = advantage

                #################
                # train cycles
                #################
                for _i in range(train_cycles):
                    # batch = memory.sample(batch_size)
                    batch = memory.sample(batch_size)

                    with tf.GradientTape() as tape:
                        batch["model_probs"], batch["model_value"] = model(batch["state"], training=True)
                        loss = loss_fn(**batch)

                    gradients = tape.gradient(loss, model_variables)
                    gradients = zip(gradients, model_variables)

                    optimizer.apply_gradients(gradients)

                    # checkpoints
                    if tf.equal(optimizer.iterations % checkpoint_steps, 0):
                        model.save_weights(
                            filepath=os.path.join(self.model_dir, "model"),
                            save_format="tf",
                        )

                if beta > 0 or actor == "target":
                    assert beta > 0

                    eager_utils.update_variables_soft(
                        target_variables,
                        model_variables,
                        beta,
                    )

                    if actor == "model":
                        eager_utils.update_variables(
                            model_variables,
                            target_variables,
                        )

                with summary_writer.as_default(), summary_ops_v2.always_record_summaries():
                    summary_ops_v2.scalar("summary/mean_target_value", memory["target_value"].mean(), step=episode)
                    summary_ops_v2.scalar("summary/mean_return", memory["returns"].mean(), step=episode)
                    summary_ops_v2.scalar("summary/mean_reward", memory["reward"].mean(), step=episode)

                memory.reset()
