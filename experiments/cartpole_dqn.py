
from __future__ import print_function

import click
from dicto import click_options_config, Dicto
import gym
import tensorrl as trl
import tensorflow as tf
import tfinterface as ti
import numpy as np
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory


def model_fn(inputs, mode, params):

    training = mode == tf.estimator.ModeKeys.TRAIN

    net = inputs["state0"]

    net = tf.layers.flatten(net)

    net = tf.layers.dense(net, 16, activation=tf.nn.relu, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=params.regularization))
    net = tf.layers.dense(net, 16, activation=tf.nn.relu, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=params.regularization))
    net = tf.layers.dense(net, 16, activation=tf.nn.relu, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=params.regularization))
    net = tf.layers.dense(net, 2, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=params.regularization))

    return net

def input_fn(env):
    state_shape = list(env.env.observation_space.shape)
    # state_shape += [env.window]

    print("STATE SHAPE")
    print(state_shape)

    return dict(
        state0 = tf.placeholder(tf.float32, shape = [None] + state_shape),
        action = tf.placeholder(tf.int32, shape = [None]),
        reward = tf.placeholder(tf.float32, shape = [None]),
        state1 = tf.placeholder(tf.float32, shape = [None] + state_shape),
        terminal = tf.placeholder(tf.bool, shape = [None]),
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

    def __getattr__(self, attr):
        return getattr(self.env, attr)


class Physics(object):

    def __init__(self, env):

        self.env = env
        self.window = 2
        self.state = None

    def reset(self, *args, **kwargs):
        new_state = self.env.reset(*args, **kwargs)
        zeros = np.zeros_like(new_state)
        self.state = np.stack([new_state, zeros], axis = -1)

        return self.state

    def step(self, *args, **kwargs):

        new_state, reward, done, info = self.env.step(*args, **kwargs)
        old_state = self.state[..., 0]

        self.state[..., 0] = new_state
        self.state[..., 1] = new_state - old_state

        return self.state, reward, done, info

    def __getattr__(self, attr):
        return getattr(self.env, attr)



@click.command()
@click.option("--model-dir", required = True)
@click.option("-v", "--visualize", is_flag = True)
@click_options_config("cartpole_dqn.yml")
def main(model_dir, visualize, **params):
    params = Dicto(params)
    print(params)

    env = gym.make('CartPole-v1')
    # env = Physics(env)

    agent = trl.agent.DQN(model_fn, model_dir, params=params)

    agent.train(
        env,
        lambda: input_fn(env),
        max_steps = params.max_steps, 
        policy = BoltzmannQPolicy(),
        memory = SequentialMemory(
            limit = params.memory_limit,
            window_length = 1,
        ),
        target_model_update = params.target_model_update,
        gamma = params.gamma,
        warmup_steps = params.warmup_steps,
        batch_size = params.batch_size,
        summary_steps = params.summary_steps,
        visualize = visualize,
    )

if __name__ == '__main__':
    main()