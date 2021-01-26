import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense


class CriticNetwork(keras.Model):
    """
    Critic network.

    :param fc1_dims: Fully connected dimensions.
    :param fc2_dims: Fully connected dimensions.
    :param name: Name.
    :param chkpt_dir: Checkpoint directory.
    """
    def __init__(self,
                 fc1_dims: int = 512,
                 fc2_dims: int = 512,
                 name: str = 'critic',
                 chkpt_dir: str = 'tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims: int = fc1_dims
        self.fc2_dims: int = fc2_dims

        self.model_name = name
        self.base_dir: str = os.path.dirname(__file__)
        self.checkpoint_dir = os.path.abspath(os.path.join(self.base_dir, chkpt_dir))
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')
        self.checkpoint_file = os.path.abspath(self.checkpoint_file)

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, *inputs, training=None, mask=None):
        """Calls the model on new input states and actions.
        This is the forward propagation operation.

        :param inputs: A tensor or list of tensors.
        :param training: Boolean or boolean scalar tensor, indicating
        whether to run the `Network` in training mode or inference mode.
        :param mask: A mask or list of masks. A mask can be either a
        tensor or None (no mask).
        :return: A tensor if there is a single output, or a list of
        tensors if there are more than one outputs.
        """
        state, action = inputs
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q

    def get_config(self):
        pass


class ActorNetwork(keras.Model):
    def __init__(self,
                 fc1_dims: int = 512,
                 fc2_dims: int = 512,
                 n_actions: int = 2,
                 name: str = 'actor',
                 chkpt_dir: str = 'tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.base_dir: str = os.path.dirname(__file__)
        self.checkpoint_dir = os.path.abspath(os.path.join(self.base_dir, chkpt_dir))
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')
        self.checkpoint_file = os.path.abspath(self.checkpoint_file)

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')

    def call(self, inputs, training=None, mask=None):
        """Calls the model on new input states.

        :param inputs: States.
        """
        # The output values of the deep neural network are not
        # probabilities in DDPG, but actual action values:
        out_val = self.fc1(inputs)
        out_val = self.fc2(out_val)

        # If the action bounds are not +/- 1, multiply here by the
        # right magnitude:
        mu = self.mu(out_val)

        return mu

    def get_config(self):
        pass


if __name__ == '__main__':
    pass