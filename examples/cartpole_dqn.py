
from __future__ import print_function

import fire
import dicto as do
import gym
import tensorrl as trl
import tensorflow as tf
import numpy as np
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy, MaxBoltzmannQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory


def model_fn(params):

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape = [4]),
        tf.keras.layers.Dense(32, activation = tf.nn.relu),
        tf.keras.layers.Dense(32, activation = tf.nn.relu),
        tf.keras.layers.Dense(32, activation = tf.nn.relu),
        tf.keras.layers.Dense(2, use_bias = False),
    ])

    return model


class API:

    @do.fire_options("examples/cartpole_dqn.yml", "params")
    def train(
        self, 
        model_dir,
        params,
        visualize = False,
        visualize_eval = False,
        ):

        print(params)

        env = gym.make('CartPole-v1')
        env._max_episode_steps = 2000

        np.random.seed(params.seed)
        env.seed(params.seed)

        agent = trl.eager.DQN(
            lambda: model_fn(params), 
            model_dir, 
            params=params,
        )

        agent.train(
            env,
            max_steps = params.max_steps,
            policy = MaxBoltzmannQPolicy(eps=0.9),
            memory = SequentialMemory(
                limit = params.memory_limit,
                window_length = 1,
            ),
            target_model_update = params.target_model_update,
            gamma = params.gamma,
            warmup_steps = params.warmup_steps,
            batch_size = params.batch_size,
            summary_steps = params.summary_steps,
            save_steps = params.save_steps,
            visualize = visualize,
            visualize_eval = visualize_eval,
            seed = params.seed,
            double_dqn = True,
            eval_episode_frequency = params.eval_episode_frequency,
            eval_episodes = params.eval_episodes,
        )



    @do.fire_options("examples/cartpole_dqn.yml", "params")
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
            max_steps = params.max_steps,
            policy = EpsGreedyQPolicy(eps=0.0),
            visualize = visualize,
            seed = params.seed,
        )

def main():
    fire.Fire(API)

if __name__ == '__main__':
    main()