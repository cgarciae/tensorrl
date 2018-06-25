
from __future__ import print_function

import click
from dicto import click_options_config, Dicto
import gym
import tensorrl as trl
import tensorflow as tf
import tfinterface as ti
import numpy as np
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy, MaxBoltzmannQPolicy, GreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

def get_model_fn(env):

    def model_fn(inputs, mode, params):

        training = mode == tf.estimator.ModeKeys.TRAIN

        net = inputs["state"]   

        net = tf.layers.flatten(net)

        net = tf.layers.dense(net, 32)
        # net = tf.layers.dense(net, 64, activation=tf.nn.relu)
        net = tf.layers.dense(net, 128, activation=tf.nn.relu)
        # net = tf.concat([net, tf.layers.dense(net, 12, activation=tf.nn.relu)], axis = 1)
        # net = tf.concat([net, tf.layers.dense(net, 12, activation=tf.nn.relu)], axis = 1)
        # net = tf.concat([net, tf.layers.dense(net, 12, activation=tf.nn.relu)], axis = 1)
        # net = tf.layers.dense(net, 16, activation=tf.nn.relu)
        # net = tf.layers.dropout(net, rate = 0.3)
        net = tf.layers.dense(net, env.action_space.n) 

        return net
    
    return model_fn

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


@click.group()
def main():
    pass

@main.command("train")
@click.option("--model-dir", required = True)
@click.option("-v", "--visualize", is_flag = True)
@click_options_config("examples/lunar_lander_dqn.yml", "params")
def train(model_dir, visualize, params):
    print(params)

    env = gym.make('LunarLander-v2')
    env._max_episode_steps = 2000

    env = trl.env.TimeExpanded(env, 3)

    np.random.seed(params.seed)
    env.seed(params.seed)

    agent = trl.prototype.DQN(get_model_fn(env), model_dir, params=params)

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
        double_dqn = True,
        train_cycles=params.train_cycles,
    )


@main.command("eval")
@click.option("--model-dir", required = True)
@click.option("-v", "--visualize", is_flag = True)
@click_options_config("examples/lunar_lander_dqn.yml", "params")
def eval(model_dir, visualize, params):
    print(params)

    env = gym.make('LunarLander-v2')
    env._max_episode_steps = 2000

    env = trl.env.TimeExpanded(env, 3)

    np.random.seed(params.seed)
    env.seed(params.seed)

    agent = trl.prototype.DQN(get_model_fn(env), model_dir, params=params)

    agent.eval(
        env,
        lambda: input_fn(env),
        max_steps = params.max_steps,
        policy = EpsGreedyQPolicy(eps=0.0),
        visualize = visualize,
        seed = params.seed,
    )

if __name__ == '__main__':
    main()