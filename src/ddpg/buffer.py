from __future__ import annotations
import sys

import numpy as np
from numpy.core import ndarray


class ReplayBuffer:
    """
    This buffer stores the states, actions, rewards, new states,
    and terminal flags that the agent encounters.

    :param max_size: Maximal size of the buffer's memory. Note
    that the memory is bounded.
    :param input_shape: Input shape from the Environment.
    :param n_actions: Number of actions we take in the
    Environment. Actions are all continuous.

    :ivar mem_size: Memory size of the buffer. Note that it is
    bounded. As a consequence, earlier experiences get overwritten
    as we exceed the memory.
    :ivar mem_cntr: Memory counter.
    """
    def __init__(self, max_size: int, input_shape: tuple[int, int], n_actions: int):
        self.mem_size: int = max_size
        self.mem_cntr: int = 0

        # Memories:
        self.state_memory: ndarray = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory: ndarray = np.zeros((self.mem_size, *input_shape))
        self.action_memory: ndarray = np.zeros((self.mem_size, n_actions))
        self.reward_memory: ndarray = np.zeros(self.mem_size)
        self.terminal_memory: ndarray = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done: bool):
        # Ensures circular behaviour when the number of experiences exceeds the memory size:
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size) -> tuple:
        # Keep track of how much of the memory is filled so that
        # we never include empty experiences in our batches:
        max_mem = min(self.mem_cntr, self.mem_size)

        # Randomly sample experiences from the memory;
        # `replace=False` prevents from double re-sampling memories:
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states: ndarray = self.state_memory[batch]
        states_: ndarray = self.new_state_memory[batch]
        actions: ndarray = self.action_memory[batch]
        rewards: ndarray = self.reward_memory[batch]
        dones: ndarray = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

    def get_size_per_attr(self):
        size_per_attr: dict = {}
        for k, v in self.__dict__.items():
            memsize: int = sys.getsizeof(v)
            memstr: str = f"{memsize} bits" if memsize < 1e6 else f"{memsize * 1e-6:.2f} MB"
            size_per_attr[k] = memstr
        return size_per_attr
