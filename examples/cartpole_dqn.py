
from __future__ import print_function

import click
from dicto import click_options_config, Dicto
import gym
import tensorrl as trl
import tensorflow as tf
import tfinterface as ti
import numpy as np
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy, MaxBoltzmannQPolicy
from rl.memory import SequentialMemory


def model_fn(inputs, mode, params):

    training = mode == tf.estimator.ModeKeys.TRAIN

    net = inputs["state"]   

    net = tf.layers.flatten(net)

    net = tf.layers.dense(net, 16, activation=tf.nn.relu) #, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=params.regularization))
    net = tf.layers.dense(net, 16) #, activation=tf.nn.relu, batch_norm=dict(training=training)) #, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=params.regularization))
    net = tf.layers.dense(net, 16) #, activation=tf.nn.relu, batch_norm=dict(training=training)) #, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=params.regularization))
    net = tf.layers.dense(net, 2, use_bias=False) #, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=params.regularization))

    return net

def input_fn(env):
    state_shape = list(env.env.observation_space.shape)
    state_shape += [env.window]

    print("STATE SHAPE")
    print(state_shape)

    return dict(
        state0 = tf.placeholder(tf.float32, shape = [None] + state_shape),
        action = tf.placeholder(tf.int32, shape = [None]),
        reward = tf.placeholder(tf.float32, shape = [None]),
        state1 = tf.placeholder(tf.float32, shape = [None] + state_shape),
        terminal = tf.placeholder(tf.bool, shape = [None]),
    )



@click.command()
@click.option("--model-dir", required = True)
@click.option("-v", "--visualize", is_flag = True)
@click_options_config("cartpole_dqn.yml", "params")
def main(model_dir, visualize, params):
    print(params)

    env = gym.make('CartPole-v1')
    env._max_episode_steps = 2000

    env = trl.env.TimeExpanded(env, 3)

    np.random.seed(params.seed)
    env.seed(params.seed)

    agent = trl.prototype.DQN(model_fn, model_dir, params=params)

    agent.train(
        env,
        lambda: input_fn(env),
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
        seed = params.seed,
    )

if __name__ == '__main__':
    main()