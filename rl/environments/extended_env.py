from abc import ABCMeta

import gym
from typing import Tuple

from numpy import ndarray
from typing import Tuple


class ExtendedEnv(gym.Env, metaclass=ABCMeta):
    def __init__(self, config_path):
        super(ExtendedEnv, self).__init__()

    action_count: int
    delta_t: float
    def log_params(self):
        raise NotImplementedError

    def _observations(self) -> Tuple[float, ...]:
        """
        Getter for the current state in the (semi) normalized observation format
        @return:
        """
        raise NotImplementedError

    def set_state(self, x_start, y_start, theta_start) -> ndarray:
        """
        Helper method to explicitly set the system state
        resets the action counter
        @param x_start: initial x position
        @param y_start: initial y position
        @param theta_start: initial orientation
        @return: observation (np.Array (x, y, theta) in observation space)
        """
        raise NotImplementedError

    def get_state(self) -> Tuple[float, ...]:
        """
        Getter method for the current state of the system
        @return:
        """
        raise NotImplementedError
