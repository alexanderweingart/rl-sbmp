import gym
import yaml
import os
import random
from gym import spaces
import numpy as np
from numpy import ndarray

from project_utils import utils
import math
import time
import logging
from dynamics import AckermannCarDynamicsFirstOrder
import rl.environments.environment_utils as env_utils
from artist.AckermannCarArtist import AckermannCarArtist
from rl.environments.extended_env import ExtendedEnv
from typing import Tuple, List, Optional


class CustomEnv(ExtendedEnv):
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 40}

    def __init__(self, config_path: str, trace_mode=False):
        super().__init__(config_path)
        self.logger = logging.getLogger(__file__)

        if config_path is None:
            self.logger.error("The given config_path is None!")
            raise ValueError

        self.config = yaml.safe_load(open(config_path))

        self.dynamics = AckermannCarDynamicsFirstOrder(config_path)

        self.goal_tolerance_pos = env_utils.get_param_from_config(self.config, "goal_tolerance_pos", self.logger)
        self.goal_tolerance_theta = env_utils.get_param_from_config(self.config, "goal_tolerance_theta", self.logger)

        self.max_steps = env_utils.get_param_from_config(self.config, "max_steps", self.logger)
        self.no_step_penalty = env_utils.get_param_from_config(self.config, "no_step_penalty", self.logger)

        self.reset_r_min = self.goal_tolerance_pos
        self.reset_r_max = 0.5

        self.artist = None
        self.trace_mode = trace_mode

        # rl params
        self.action_space_shape = (2,)
        self.action_space_low = -1
        self.action_space_high = 1
        self.action_space_dtype = np.float32

        self.observation_space_shape = (3,)
        self.observation_space_low = -1
        self.observation_space_high = 1
        self.observation_space_dtype = np.float32

        self.action_space = spaces.Box(
            low=self.action_space_low,
            high=self.action_space_high,
            shape=self.action_space_shape,
            dtype=self.action_space_dtype
        )  # (v, phi)

        self.observation_space = spaces.Box(
            low=self.observation_space_low,
            high=self.observation_space_high,
            shape=self.observation_space_shape,
            dtype=self.observation_space_dtype
        )  # (x_rel, y_rel, theta_rel)

        # state initialisation
        self.action_count = 0
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.x_min, self.x_max = -1, 1
        self.y_min, self.y_max = -1, 1
        self.terminated = False

        self.last_x = self.x
        self.last_y = self.y
        self.last_theta = self.theta

        self.version_id = f"ackermann_car_vel_ctrl"

        if self.no_step_penalty:
            self.version_id += "_no_step_penalty"

        self.reset()

        self.logger.info(f"environment initialised")
        self.log_params()

    def log_params(self):
        self.logger.info(f"## action space ##")
        self.logger.info(f"shape: {self.action_space_shape}")
        self.logger.info(f"min: {self.action_space_low} max: {self.action_space_high}")
        self.logger.info(f"dtype: {self.action_space_dtype}")
        self.logger.info(f"## observation space ##")
        self.logger.info(f"shape: {self.observation_space_shape}")
        self.logger.info(f"min: {self.observation_space_low}")
        self.logger.info(f"max: {self.observation_space_high}")
        self.logger.info(f"dtype: {self.observation_space_dtype}")
        self.logger.info(f"## goal tolerance ##")
        self.logger.info(f"theta: {self.goal_tolerance_theta}")
        self.logger.info(f"pos: {self.goal_tolerance_pos}")
        self.logger.info(f"## rl config ##")
        self.logger.info(f"max_steps: {self.max_steps}")
        self.logger.info(f"reset radius between {self.reset_r_min} and {self.reset_r_max}")
        self.logger.info("## dynamics ##")
        self.logger.info(self.dynamics.get_param_string())

    def _reward(self) -> Tuple[float, bool]:
        """
        Method to calculuate the reward of the current state
        @return: reward, terminated
        """
        step_penalty = 0.0 if self.no_step_penalty else -(1.0 / self.max_steps)
        d = math.dist([self.x, self.y], [0.0, 0.0])
        d_o = min(abs(self.theta), 2 * math.pi - abs(self.theta))

        if d < self.goal_tolerance_pos and d_o < self.goal_tolerance_theta:  # target reached
            return 1.0, True

        max_steps_reached = self.action_count >= self.max_steps - 1
        return step_penalty, max_steps_reached

    def _observations(self) -> Tuple[float, float, float]:
        """
        Getter for the current state in the (semi) normalized observation format
        @return:
        """
        x_obs = self.x
        y_obs = self.y
        theta_obs = utils.map_to_diff_range(0, np.pi * 2, self.observation_space_low, self.observation_space_high,
                                            self.theta)

        return x_obs, y_obs, theta_obs

    def step(self, action: List[float]) -> Tuple[ndarray, float, bool, dict]:
        """
        Transition function of the environment
        @param action: the action that should be enacted in the environment. (in the normalized range)
        @return: observation (np.Array (x, y, theta) in observation range), reward, terminated, info (empty dict)
        """
        lin_vel, phi = self.dynamics.convert_normalized_action(action)

        self.last_x = self.x
        self.last_y = self.y
        self.last_theta = self.theta

        self.x, self.y, self.theta = self.dynamics.step([self.x, self.y, self.theta], [lin_vel, phi])

        x_obs, y_obs, theta_obs = self._observations()

        reward, terminated = self._reward()
        self.action_count += 1
        self.terminated = terminated

        observation = np.array([x_obs, y_obs, theta_obs], dtype=np.float32)

        return observation, reward, terminated, {}

    def reset(self) -> ndarray:
        """
        resets the system to random new state
        sampling is done via polar coordinate sampling (r and phi get selected randomly and x,y are constructed accordingly)
        r is restricted by the class parameters reset_r_min and reset_r_max
        @return: np.Array (x, y, theta) (all in observation state)
        """
        r = utils.map_to_diff_range(0, 1, self.reset_r_min, self.reset_r_max, np.random.random())
        phi = 2 * math.pi * random.random()

        self.x = math.cos(phi) * r
        self.y = math.sin(phi) * r
        self.theta = 2 * math.pi * random.random()

        self.last_x = self.x
        self.last_y = self.y
        self.last_theta = self.theta

        self.action_count = 0

        x_obs, y_obs, theta_obs = self._observations()

        return np.array([x_obs, y_obs, theta_obs], dtype=np.float32)

    def set_state(self, x_start, y_start, theta_start) -> ndarray:
        """
        Helper method to explicitly set the system state
        resets the action counter
        @param x_start: initial x position
        @param y_start: initial y position
        @param theta_start: initial orientation
        @return: observation (np.Array (x, y, theta) in observation space)
        """
        self.x = x_start
        self.y = y_start
        self.theta = theta_start

        self.last_x = x_start
        self.last_y = y_start
        self.last_theta = theta_start

        (x_obs, y_obs, theta_obs) = self._observations()
        observation = np.array([x_obs, y_obs, theta_obs], dtype=np.float32)

        self.action_count = 0

        return observation

    def get_state(self) -> Tuple[float, float, float]:
        """
        Getter method for the current state of the system
        @return:
        """
        return self.x, self.y, self.theta

    def render(self, mode="human", perfect_path=None, overlay_traces=False, screenshot_dir=None) -> Optional[ndarray]:
        """
        Uses the AckermannCarArtist to render the current state of the system
        @param mode: rendering mode. see the metadata for provided modes
        @param perfect_path: the perfect path from init to goal. if this is provided, it will be drawn as a slim grey line
        @param overlay_traces: show the previous rollouts as well?
        @param screenshot_dir: path to a directory where screenshots should be stored (if desired)
        @return: None or ndarray if the rendering mode is set to rgb_array
        """
        if self.artist is None:
            self.artist = AckermannCarArtist(car_length=self.dynamics.car_length,
                                             render_mode="human",  # only rendering mode available for now
                                             render_fps=self.metadata["render_fps"],
                                             trace_mode=self.trace_mode,
                                             screen_width=500,
                                             screen_height=500,
                                             x_max=1
                                             )

        first_state = True if self.action_count == 0 else False  # is this the first state?
        last_state = self.terminated  # is this the last state? a.k.a. is this env terminated?
        return self.artist.render_environment_state(self.x, self.y, self.theta,
                                                    self.goal_tolerance_pos, self.goal_tolerance_theta,
                                                    perfect_path, overlay_traces, screenshot_dir, first_state,
                                                    last_state,
                                                    self.last_x, self.last_y)

    def close(self):
        print(f"closing now")
