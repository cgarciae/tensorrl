
from __future__ import print_function

import click
from dicto import click_options_config, Dicto
import gym
import tensorrl as trl
import tensorflow as tf
import numpy as np
import rl as krl


def model_fn(inputs, mode, params):

    net = inputs["state0"]

    net = tf.layers.flatten(net)

    net = tf.layers.dense(100, activation=tf.nn.relu)
    net = tf.layers.dense(2)

    return net

def input_fn(env):
    state_shape = env.observation_space.shape

    return dict(
        state0 = tf.placeholder(tf.float32, shape = [None] + state_shape),
        action = tf.placeholder(tf.int32, shape = [None, 1]),
        reward = tf.placeholder(tf.float32, shape = [None, 1]),
        state1 = tf.placeholder(tf.float32, shape = [None] + state_shape),
        terminal = tf.placeholder(tf.bool, shape = [None, 1]),
    )

class TimeExpanded(object):

    def __init__(self, env, window):

        self.env = env
        self.window = window
        self.state = None

    def reset(self, *args, **kwargs):
        new_state = self.env.reset(*args, **kwargs)
        self.state = np.stack([new_state] * self.window, axis = -1)

        return self.state

    def step(self, *args, **kwargs):

        new_state, reward, done, info = self.env.step(*args, **kwargs)

        self.state[..., 1:] = self.state[..., :-1]
        self.state[..., 0] = new_state

        return self.state, reward, done, info


@click.command()
@click.option("--model-dir", required = True)
@click_options_config("cartpole_dqn.yml")
def main(model_dir, **params):
    params = Dicto(params)

    env = gym.make('CartPole-v1')
    env = TimeExpanded(env, 4)

    agent = trl.agent.dqn_proto.DQN(model_fn, model_dir, params=params)

    agent.train(
        env,
        lambda: input_fn(env),
        max_steps = params.max_steps, 
        policy = krl.policy.EpsGreedyQPolicy(
            eps = params.eps,
        ),
        memory = krl.memory.SequentialMemory(
            limit = params.memory_limit,
        ),
        target_model_update = params.target_model_update,
        gamma = params.gamma,
        warmup_steps = params.warmup_steps,
        batch_size = params.batch_size,
        summary_steps = params.summary_steps,
    )

if __name__ == '__main__':
    main()