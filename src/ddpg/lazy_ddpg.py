from __future__ import annotations
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gym
from typing import Optional


def ddpg(env_fn,
         ac_kwargs: Optional[dict] = None,
         seed: int = 0,
         save_folder: Optional[str] = None,
         num_train_episodes: int = 100,
         test_agent_every: int = 25,
         replay_size: int = int(1e6),
         gamma: float = 0.99,
         decay: float = 0.995,
         mu_lr: float = 1e-3,
         q_lr: float = 1e-3,
         batch_size: int = 100,
         start_steps: int = 10_000,
         action_noise: float = 0.1,
         max_episode_length: int = 1_000):
    """
    DDPG algorithm implementation.

    :param env_fn: Callable function that returns
    an instance of the environment.
    :param ac_kwargs: Actor-Critic keyword arguments
    that are passed to neural networks. Used to
    specify hidden layer sizes, activation functions,
    and more.
    :param seed: Seed to make experiments reproducible.
    :param save_folder: Path specifying where to
    save videos of the agent playing the environment.
    :param num_train_episodes: Number of episodes used
    for training.
    :param test_agent_every: Specify after how many
    training episodes we should test the agent and
    save a video playing 25 episodes.
    :param replay_size: Specify the size of the
    experience replay buffer.
    :param gamma: The discount factor for calculating
    the return.
    :param decay: The smoothing constant to use when
    we update the target network parameters.
    :param mu_lr: Policy network learning rate.
    :param q_lr: Value network learning rate.
    :param batch_size: Specify the batch size to
    sample from the experience replay buffer.
    :param start_steps: Specify after how many steps
    from the beginning of training we should perform
    random actions without using the Policy Network μ.
    This enables exploration and also filling up the
    experience replay buffer as we do in a regular
    deep Q-learning, as well.
    :param action_noise: The scale of the gaussian
    to sample from when adding exploration noise
    to the Policy network output.
    :param max_episode_length: Specify the maximum
    length of an episode to go through before exiting.
    :return:
    """
    if ac_kwargs is None:
        ac_kwargs = {}

    tf.random.set_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()

    # To record a video of the agent:
    if save_folder is not None:
        test_env = gym.wrappers.Monitor(test_env, save_folder)

    num_states: int = env.observation_space.shape[0]
    num_actions: int = env.action_space.shape[0]

    # Maximum value of action
    # Assumes both low and high values are the same
    # Assumes all actions have the same bounds
    # May NOT be the case for all environments
    action_max = env.action_space.high[0]

    # Create Tensorflow 2.x variables (neural network inputs)
    # TODO: What are these tensors used for and how to create them in TF2?
    X = tf.Variable(tf.zeros(shape=(num_states,)), dtype=tf.float32)  # state
    A = tf.Variable(tf.zeros(shape=(num_actions,)), dtype=tf.float32)  # action
    X2 = tf.Variable(tf.zeros(shape=(num_states,)), dtype=tf.float32)  # next state
    R = tf.Variable(tf.zeros(shape=(1,)), dtype=tf.float32)  # reward
    D = tf.Variable(tf.zeros(shape=(1,)), dtype=tf.float32)  # done

    mu, q, q_mu = create_networks(X, A, num_actions, action_max, **ac_kwargs)
    # We use X2 because we want to maximize max_a{ Q(s', a) }
    # Where this is equal to Q(s', μ(s'))
    # This is because it's used in the target calculation: r + gamma * (1 - d) * max_a{ Q(s',a) }
    _, _, q_mu_target = create_networks(X2, A, num_actions, action_max, **ac_kwargs)

    replay_buffer = ReplayBuffer(obs_dim=num_states, act_dim=num_actions, size=replay_size)

    # Target value for the Q network loss
    # This weights should never be differentiated, thus, we should not
    # use backpropagation on them. These weights should never be learnt, rather
    # we just have to update them using a moving average.
    q_target = tf.stop_gradient(R + gamma * (1 - D) * q_mu_target)

    # DDPG losses
    mu_loss = -tf.reduce_mean(q_mu)
    q_loss = tf.reduce_mean((q - q_target) * (q - q_target))

    # Train each network separately
    mu_optimizer = Adam(learning_rate=mu_lr)
    q_optimizer = Adam(learning_rate=q_lr)
    # We want to maximize Q wrt μ
    mu_train_op = mu_optimizer.minimize(mu_loss, var_list=[mu])
    q_train_op = q_optimizer.minimize(q_loss, var_list=[q])

    # Use soft updates to update the target networks
    target_update = tf.group(
        [tf.assign(v_targ, decay * v_targ + (1 - decay) * v_main)
         for v_main, v_targ in zip(get_vars('main'), q_mu_target)]
    )


class ReplayBuffer:
    """
    The experience replay memory.
    """
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.pointer, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        p: int = self.pointer
        self.obs1_buf[p] = obs
        self.obs2_buf[p] = next_obs
        self.acts_buf[p] = act
        self.rews_buf[p] = rew
        self.done_buf[p] = done
        # Ensures circular appending:
        self.pointer = (p + 1) % self.max_size
        # True size (i.e. with filled fields) of the buffer
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(s=self.obs1_buf[idxs],
                    s2=self.obs2_buf[idxs],
                    a=self.acts_buf[idxs],
                    r=self.rews_buf[idxs],
                    d=self.done_buf[idxs])


def ann(x,
        layer_sizes,
        hidden_activation=tf.nn.relu,
        output_activation=None):
    """
    Create a simple feedforward neural network.

    :param x:
    :param layer_sizes:
    :param hidden_activation:
    :param output_activation:
    :return: The output node as a tensorflow.Variable.
    """
    # Create all the hidden layers:
    for h in layer_sizes[:-1]:
        x = Dense(units=h, activation=hidden_activation)(x)
    return Dense(units=layer_sizes[-1], activation=output_activation)(x)


def create_networks(s,
                    a,
                    num_actions,
                    action_max,
                    hidden_sizes=(300,),
                    hidden_activation=tf.nn.relu,
                    output_activation=tf.tanh) -> tuple:
    """
    Create both the actor and the critic network at once.
    Q[s, μ(s)] returns the maximum Q for a given state s.

    :param s: A batch of states.
    :param a: A batch of actions.
    :param num_actions: Dimensionality of the action space.
    :param action_max: Maximum absolute value of the actions.
    This assumes that the lower and the upper bounds are
    symmetric.
    :param hidden_sizes: Size of the hidden layers of each
    network.
    :param hidden_activation: Hidden activation function.
    :param output_activation: Output activation that refers
    to the policy network. The reason is that the Q network
    has no output activation since it outputs a real number
    representing future rewards.
    :return: Two neural networks.
    """
    ls1 = list(hidden_sizes) + [num_actions]
    # μ is between [-action_max, +action_max] since the output
    # activation is tanh [-1, +1]:
    mu = action_max * ann(s, ls1, hidden_activation, output_activation)

    input_q = tf.concat([s, a], axis=-1)  # (state, action)
    ls2 = list(hidden_sizes) + [1]
    # Get rid of redundant dimensions
    q = tf.squeeze(ann(input_q, ls2, hidden_activation, None), axis=1)

    input_q_mu = tf.concat([s, mu], axis=-1)  # (state, action)
    ls2 = list(hidden_sizes) + [1]
    # Get rid of redundant dimensions
    q_mu = tf.squeeze(ann(input_q_mu, ls2, hidden_activation, None), axis=1)

    return mu, q, q_mu
