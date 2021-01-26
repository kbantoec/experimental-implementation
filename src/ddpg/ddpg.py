from __future__ import annotations
import numpy as np
from numpy.core import ndarray
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from src.ddpg.buffer import ReplayBuffer
from src.ddpg.networks import ActorNetwork, CriticNetwork
from typing import Optional, Union


class Agent:
    def __init__(self,
                 input_dims: tuple[int, int],
                 alpha: float = 1e-3,
                 beta: float = 2e-3,
                 env=None,
                 gamma: float = 0.99,
                 n_actions: int = 2,
                 max_size: int = 1_000_000,
                 tau: float = 5e-3,
                 fc1: int = 400,
                 fc2: int = 300,
                 batch_size: int = 64,
                 noise: float = 0.1):
        """
        :param input_dims:
        :param alpha: Learning rate for the actor network.
        :param beta: Learning rate for the critic network.
        :param env: The environment. The max and the min
        actions are needed because we want to integrate
        some noise into the output of the deep neural
        network for some exploration to be possible.
        :param gamma: Discount factor for the update
        equation.
        :param n_actions: The number of actions.
        :param max_size: Maximal memory size for the
        replay buffer.
        :param tau:
        :param fc1: Number of units for fully connected
        hidden layer 1.
        :param fc2: Number of units for fully connected
        hidden layer 2.
        :param batch_size: Batch size for memory sampling.
        :param noise: The standard deviation of the
        noise sampled from a gaussian distribution. The
        included noise is used for the exploration.
        """
        self.gamma: float = gamma
        self.tau: float = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size: int = batch_size
        self.n_actions: int = n_actions
        self.noise: float = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = ActorNetwork(n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(name='critic')
        self.target_actor = ActorNetwork(n_actions=n_actions, name='target_actor')
        self.target_critic = CriticNetwork(name='target_critic')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        # We actually do not call an update to the target networks, but
        # tensorflow imposes us this step to perform our soft copy:
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        # Hard copy of the initial copy of the target networks:
        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau: Optional[Union[int, float]] = None):
        """Do hard (if tau == 1) or soft copies of the actor and critic network
        to update the weights of their respective target networks."""
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, observation, training: bool = True) -> ndarray:
        """

        :param observation: An observation from the environment.
        :param training: Flag for training or testing/evaluating.
        :return:
        """
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if training:
            actions += tf.random.normal(shape=[self.n_actions],
                                        mean=0.0, stddev=self.noise)
        # note that if the environment has an action > 1, we have to multiply by
        # max action at some point
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]

    def learn(self):
        # If the batch size is smaller than the memory stored
        # in the buffer, then do not sample the memories, i.e.,
        # do not learn:
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            # We have to squeeze the sample of batch dimension because otherwise it
            # does not learn:
            critic_value_ = tf.squeeze(self.target_critic(states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = reward + self.gamma * critic_value_ * (1 - done)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient,
                                                  self.critic.trainable_variables))

        # Based on current set of weights
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            # Negative because we are doing gradient ascent since we want
            # to maximize the total return over time:
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient,
                                                 self.actor.trainable_variables))

        self.update_network_parameters()
