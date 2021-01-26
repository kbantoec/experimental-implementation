import gym
import numpy as np
import multiprocessing
import tensorflow as tf
import tensorflow.keras.layers as kl


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        # Sample a random categorical action from the given logits.
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        # Note: no tf.get_variable(), just simple Keras API!
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation='relu')
        self.value = kl.Dense(1, name='value')
        # Logits are unnormalized log probabilities.
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        x = tf.convert_to_tensor(inputs)
        # Separate hidden layers from the same input tensor.
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, obs):
        # Executes `call()` under the hood.
        logits, value = self.predict_on_batch(obs)
        action = self.dist.predict_on_batch(logits)
        # Another way to sample actions:
        #   action = tf.random.categorical(logits, 1)
        # Will become clearer later why we don't use it.
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


class A2CAgent:
    def __init__(self, model):
        self.model = model

    def test(self, env, render=True):
        obs, done, episode_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward_t, done, _ = env.step(action)
            episode_reward += reward_t
            if render:
                env.render()
            return episode_reward


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    model = Model(num_actions=env.action_space.n)

    obs = env.reset()
    action, value = model.action_value(obs[None, :])
    print(action, value)  # [1] [-0.00145713]

    agent = A2CAgent(model)
    rewards_sum = agent.test(env)
    print("%d out of 200" % rewards_sum)  # 18 out of 200
