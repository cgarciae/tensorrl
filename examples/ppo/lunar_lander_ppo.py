from __future__ import print_function

import fire
import dicto as do
import gym
import tensorrl as trl
import tensorflow as tf
import numpy as np
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy, MaxBoltzmannQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory

PARAMS = dict(
    gamma=0.99,
    lambda_=0.99,
    batch_size=8,
    summary_steps=100,
    save_steps=10000,
    max_steps=1000000,
    learning_rate=0.001,
    seed=123,
    eval_episode_frequency=20,
    eval_episodes=2,
    epsilon=0.3,
    eps=0.5,
    horizon=8,
    epochs=10,
    critic_loss_type="clipped",
    huber_delta=100.0,
    beta=0.95,
    actor="model",
)


class ActorCritic(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        input_layer = tf.keras.layers.InputLayer(input_shape=[8])

        self.actor = tf.keras.Sequential([
            input_layer,
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            # tf.keras.layers.Dense(256, activation = tf.nn.relu),
            tf.keras.layers.Dense(4, use_bias=False, activation="softmax"),
        ])
        self.critic = tf.keras.Sequential([
            input_layer,
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            # tf.keras.layers.Dense(256, activation = tf.nn.relu),
            tf.keras.layers.Dense(1),
        ])

    def call(self, state, training=True):

        probs = self.actor(state, training=training)
        value = self.critic(state, training=training)

        return probs, value


def model_fn(params):
    return ActorCritic()


class API:
    @do.fire_options(PARAMS, "params")
    def train(
            self,
            model_dir,
            params,
            visualize=False,
            visualize_eval=False,
    ):

        print(params)

        env = gym.make('LunarLander-v2')
        env._max_episode_steps = 1000

        agent = trl.eager.PPO(
            lambda: model_fn(params),
            model_dir,
        )

        agent.train(
            env,
            max_steps=params.max_steps,
            policy=trl.policy.BoltzmannActorPolicy(from_logits=False),
            memory=trl.memory.ReplayMemory(max_size=params.horizon),
            gamma=params.gamma,
            batch_size=params.batch_size,
            summary_steps=params.summary_steps,
            save_steps=params.save_steps,
            visualize=visualize,
            visualize_eval=visualize_eval,
            seed=params.seed,
            double_dqn=False,
            eval_episode_frequency=params.eval_episode_frequency,
            eval_episodes=params.eval_episodes,
            epsilon=params.epsilon,
            lambda_=params.lambda_,
            train_cycles=int((float(params.horizon) / params.batch_size) * params.epochs),
            horizon=params.horizon,
            critic_loss_type=params.critic_loss_type,
            huber_delta=params.huber_delta,
            beta=params.beta,
            actor=params.actor,
        )

    @do.fire_options("examples/dqn/cartpole_dqn.yml", "params")
    def eval(
            self,
            model_dir,
            visualize,
            params,
    ):

        print(params)

        env = gym.make('CartPole-v1')
        env._max_episode_steps = 2000

        env = trl.env.TimeExpanded(env, 3)

        np.random.seed(params.seed)
        env.seed(params.seed)

        agent = trl.prototype.DQN(model_fn, model_dir, params=params)

        agent.eval(
            env,
            lambda: input_fn(env),
            max_steps=params.max_steps,
            policy=EpsGreedyQPolicy(eps=0.0),
            visualize=visualize,
            seed=params.seed,
        )


def main():
    fire.Fire(API)


if __name__ == '__main__':
    main()