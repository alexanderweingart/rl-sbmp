import logging
from typing import List, Optional
from numpy import ndarray

from dynamics import AckermannCarDynamicsSecondOrder
import gym
import pathlib
import yaml
import os
import random
from gym import spaces
import numpy as np
from project_utils.utils import map_to_diff_range
import math
import time
import rl.environments.environment_utils as env_utils
from artist.AckermannCarArtist import AckermannCarArtist
from typing import Tuple


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human"], "render_fps": 10}

    def __init__(self, config_path, trace_mode=False):
        super(CustomEnv, self).__init__()
        self.logger = logging.getLogger(__file__)

        if config_path is None:
            self.logger.error("The given config_path is None!")
            raise ValueError

        self.config = yaml.safe_load(open(config_path))
        self.dynamics = AckermannCarDynamicsSecondOrder(config_path)


        self.goal_tolerance_theta = env_utils.get_param_from_config(self.config, "goal_tolerance_theta", self.logger)
        self.goal_tolerance_pos = env_utils.get_param_from_config(self.config, "goal_tolerance_pos", self.logger)
        self.goal_tolerance_lin_vel \
            = env_utils.get_param_from_config(self.config, "goal_tolerance_lin_vel", self.logger)
        self.goal_tolerance_phi = env_utils.get_param_from_config(self.config, "goal_tolerance_phi", self.logger)

        self.max_steps = env_utils.get_param_from_config(self.config, "max_steps", self.logger)

        self.easy_training = env_utils.get_param_from_config(self.config, "easy_training", self.logger)

        self.std_noise = env_utils.get_param_from_config(self.config, "std_noise", self.logger)

        self.id_postfix = env_utils.get_param_from_config(self.config, "id_postfix", self.logger)

        self.m_d = env_utils.get_param_from_config(self.config, "m_d", self.logger)
        self.m_theta = env_utils.get_param_from_config(self.config, "m_theta", self.logger)
        self.m_v = env_utils.get_param_from_config(self.config, "m_v", self.logger)
        self.m_phi = env_utils.get_param_from_config(self.config, "m_phi", self.logger)

        self.stop_at_lin_constraint_violation = \
            env_utils.get_param_from_config(self.config, "stop_at_lin_constraint_violation", self.logger)

        self.reward_func = env_utils.get_param_from_config(self.config, "reward_func", self.logger)
        self.m_lin_vel_pen = env_utils.get_param_from_config(self.config, "m_lin_vel_pen", self.logger)

        if self.easy_training:
            training_set_path = env_utils.get_param_from_config(self.config, "training_set_path", self.logger)
            # easy training activated, load training set from path
            if training_set_path is None:
                logging.warning(
                    f"easy training is activated, but no training set path is provided! add path with the config key training_set_path")
                raise ValueError

            file_dir = pathlib.Path(__file__).parent.resolve()
            training_set_full_path = os.path.join(file_dir, "..", training_set_path)

            self.training_set = yaml.safe_load(open(training_set_full_path, "r"))["training_set"]

            if "max_n_starting_configs" in self.config:
                # if max n is given, limit starting configs to this number
                max_n_starting_configs = self.config["max_n_starting_configs"]
                self.training_set = self.training_set[:max_n_starting_configs]

            logging.warning(f"using easy training set (n: {len(self.training_set)}):")

        self.available_reward_functions = ["mixed_dense", "dist_reward_only", "step_penalty_target_reward",
                                           "target_reward_only", "terminal_dist_reward_only",
                                           "terminal_normalized_dist_reward_only", "normalized_dist_reward_only",
                                           "normalized_dist_reward_lin_vel_penalty"]

        if self.reward_func not in self.available_reward_functions:
            logging.warning(f"reward func {self.reward_func} not in set of available functions")
            raise ValueError

        self.r_step_max = 1 / self.max_steps

        self.max_dist_pos = 1.0  # d_max

        self.x_max = 1.0
        self.x_min = -1.0
        self.y_max = 1.0
        self.y_min = -1.0

        # for penalty normalisation purposes
        self.d_max = math.dist([max(abs(self.x_max), abs(self.x_min)), max(abs(self.y_max), abs(self.y_min))], [0, 0])
        self.d_o_max = math.pi
        self.max_lin_vel_diff = abs(self.dynamics.lin_vel_max - self.dynamics.lin_vel_min)
        self.max_phi_diff = abs(self.dynamics.phi_max - self.dynamics.phi_min)

        self.trace = []
        self.trace_starts = []
        self.trace_ends = []

        self.terminated = False

        self.screen = None
        self.clock = None

        self.trace_mode = trace_mode
        self.render_mode = "human"

        self.screen_width = 500
        self.screen_height = 500
        self.trace_mode = trace_mode

        self.action_count = 0

        self.action_space_shape = (2,)
        self.action_space_low = -1
        self.action_space_high = 1
        self.action_space_dtype = np.float32

        self.observation_space_shape = (7,)
        self.observation_space_low = -1
        self.observation_space_high = 1
        self.observation_space_dtype = np.float32

        self.action_space = spaces.Box(
            low=self.action_space_low,
            high=self.action_space_high,
            shape=self.action_space_shape,
            dtype=self.action_space_dtype
        )  # (v', phi')

        self.observation_space = spaces.Box(
            low=self.observation_space_low,
            high=self.observation_space_high,
            shape=self.observation_space_shape,
            dtype=self.observation_space_dtype
        )  # (x_rel, y_rel, theta_rel, v, phi, v_target, phi_target)

        # state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.lin_vel = 0.0
        self.phi = 0.0
        self.phi_target = 0.0
        self.lin_vel_target = 0.0

        self.lin_vel_obs_min = self.dynamics.lin_vel_min + self.dynamics.lin_accl_min
        self.lin_vel_obs_max = self.dynamics.lin_vel_max + self.dynamics.lin_accl_max

        self.last_x = self.x
        self.last_y = self.y
        self.last_theta = self.theta

        self.reset_r_min = self.goal_tolerance_pos
        self.reset_r_max = 0.5

        self.version_id = f"ackermann_car_accl_ctrl{self.id_postfix}_{self.reward_func}"

        if self.dynamics.clip_lin_vel:
            self.version_id += "_vclip"

        self.reset()

        logging.info(f"dubin_vel_control environment initialised")
        self.log_params()
        self.artist = None

    def log_params(self):
        logging.info(f"## action space ##")
        logging.info(f"shape: {self.action_space_shape}")
        logging.info(f"min: {self.action_space_low} max: {self.action_space_high}")
        logging.info(f"dtype: {self.action_space_dtype}")
        logging.info(f"## observation space ##")
        logging.info(f"shape: {self.observation_space_shape}")
        logging.info(f"min: {self.observation_space_low}")
        logging.info(f"max: {self.observation_space_high}")
        logging.info(f"dtype: {self.observation_space_dtype}")
        logging.info(f"## goal tolerance ##")
        logging.info(f"theta: {self.goal_tolerance_theta}")
        logging.info(f"pos: {self.goal_tolerance_pos}")
        logging.info(f"## others ##")
        logging.info(f"max_steps: {self.max_steps}")
        logging.info(f"reset radius between {self.reset_r_min} and {self.reset_r_max}")
        logging.warning(f"easy training: {'activated' if self.easy_training else 'deactivated'}")
        logging.info(f"## dynamics ##")
        logging.info(self.dynamics.get_param_string())

    def get_diffs(self):
        """
        Helper function for returning the difference between each state variable and its target
        @return: pos_diff, theta_diff, lin_vel_diff, phi_diff
        """
        diff_lin_vel = abs(self.lin_vel_target - self.lin_vel)
        diff_phi = abs(self.phi_target - self.phi)
        d_o = min(abs(self.theta), 2 * math.pi - abs(self.theta))
        d = math.dist((self.x, self.y), (0, 0))
        return d, d_o, diff_lin_vel, diff_phi

    def _reward(self):
        """
        Multiplexer method that executes the reward function specified by the class parameter reward_func
        """
        if self.reward_func == "dist_reward_only":
            return self._dist_reward_only()
        elif self.reward_func == "normalized_dist_reward_only":
            return self._normalized_dist_reward_only()
        elif self.reward_func == "step_penalty_target_reward":
            return self._step_penalty_target_reward()
        elif self.reward_func == "target_reward_only":
            return self._target_reward_only()
        elif self.reward_func == "terminal_dist_reward_only":
            return self._terminal_dist_reward_only()
        elif self.reward_func == "terminal_normalized_dist_reward_only":
            return self._terminal_normalized_dist_reward_only()

    def _target_reward_only(self) -> Tuple[float, bool]:
        """
        Returns 0 for every step other than the goal reaching one
        @return: reward (0 or 1), terminated (true if goal or the maximum nr. of steps are reached)
        """
        d, d_o, diff_lin_vel, diff_phi = self.get_diffs()
        if d < self.goal_tolerance_pos \
                and d_o < self.goal_tolerance_theta \
                and diff_lin_vel < self.goal_tolerance_lin_vel \
                and diff_phi < self.goal_tolerance_phi:  # target reached
            return 1, True

        max_steps_reached = self.action_count >= self.max_steps - 1
        return 0, max_steps_reached

    def _dist_reward_only(self) -> Tuple[float, bool]:
        f"""
        This reward is based on a non-normalized distance sum (square root of the quadratic sum of the components)
        This dense reward is returned at every step and only terminates when the max steps are reached
        @return: dense reward [0,1], terminated (true if max steps are reached)
        """
        _, d_o, diff_lin_vel, diff_phi = self.get_diffs()
        dist_sum = math.sqrt(self.x ** 2
                             + self.y ** 2
                             + d_o ** 2
                             + diff_lin_vel ** 2
                             + diff_phi ** 2)

        r = (1 - (1 / (1 + (1 / dist_sum)))) * (1 / self.max_steps)

        max_steps_reached = self.action_count >= self.max_steps - 1
        return r, max_steps_reached

    def _normalized_dist_reward_only(self) -> Tuple[float, bool]:
        """
        improved version of dist_reward_only
        components in the dist function are normalized by their max expected impact and weighted according to
        config params:
        self.m_d : distance to the target
        self.m_v : difference between lin_vel and lin_vel_target
        self.m_theta: orientation difference
        self.m_phi: steering angle difference

        @return: dense reward [0,1], terminated (if in goal region or max steps reached)
        """
        d, d_o, diff_lin_vel, diff_phi = self.get_diffs()

        # normalize the impact of each diff on the distance sum
        d_norm = d / self.d_max
        diff_lin_vel_norm = diff_lin_vel / self.max_lin_vel_diff
        diff_phi_norm = diff_phi / self.max_phi_diff
        d_o_norm = d_o / self.d_o_max

        dist_sum = self.m_d * d_norm + self.m_v * diff_lin_vel_norm + self.m_phi * diff_phi_norm + self.m_theta * d_o_norm
        r = (1 - dist_sum / (
                self.m_d + self.m_v + self.m_phi + self.m_theta)) * self.r_step_max  # r_{step} \in [0-r_{{step}_{max}]
        r = max(0, r)  # clip at 0

        if d < self.goal_tolerance_pos \
                and d_o < self.goal_tolerance_theta \
                and diff_lin_vel < self.goal_tolerance_lin_vel \
                and diff_phi < self.goal_tolerance_phi:  # target reached
            n_remaining = self.max_steps - self.action_count
            # design-descision: rewards all states in the target region the sam
            return self.r_step_max * n_remaining, True

        if self.action_count >= self.max_steps - 1:  # max steps reached
            return r, True

        return r, False

    def _terminal_dist_reward_only(self) -> Tuple[float, bool]:
        """
        same reward calculation as _dist_reward_only, but instead of receiving a dense reward at each step, the agent receives a final reward (after max steps or when reaching the goal region) based on the distance sum
        config parameter:
        self.m_d: weights the x and y components
        """

        d, d_o, diff_lin_vel, diff_phi = self.get_diffs()

        if (d < self.goal_tolerance_pos
            and d_o < self.goal_tolerance_theta
            and diff_lin_vel < self.goal_tolerance_lin_vel
            and diff_phi < self.goal_tolerance_phi) \
                or (self.action_count >= self.max_steps - 1):  # target reached or max steps

            dist_sum = math.sqrt(
                self.m_d * self.x ** 2
                + self.m_d * self.y ** 2
                + d_o ** 2
                + diff_lin_vel ** 2
                + diff_phi ** 2)
            r = (1 - (1 / (1 + (1 / dist_sum))))
            return r, True

        return 0, False

    def _terminal_normalized_dist_reward_only(self):
        """
        same reward calculation as _dist_reward_only, but instead of receiving a dense reward at each step, the agent receives a final reward (after max steps or when reaching the goal region) based on the distance sum
        config parameter:
        self.m_d: weights the x and y components
        """
        d, d_o, diff_lin_vel, diff_phi = self.get_diffs()

        d_norm = d / self.d_max
        diff_lin_vel_norm = diff_lin_vel / self.max_lin_vel_diff
        diff_phi_norm = diff_phi / self.max_phi_diff
        d_o_norm = d_o / self.d_o_max

        if (d < self.goal_tolerance_pos
            and d_o < self.goal_tolerance_theta
            and diff_lin_vel < self.goal_tolerance_lin_vel
            and diff_phi < self.goal_tolerance_phi) \
                or (self.action_count >= self.max_steps - 1):  # target reached or max steps
            dist_sum = self.m_d * d_norm \
                       + self.m_v * diff_lin_vel_norm \
                       + self.m_phi * diff_phi_norm \
                       + self.m_theta * d_o_norm

            r = 1 - dist_sum / (self.m_d + self.m_v + self.m_phi + self.m_theta)
            return r, True

        return 0, False

    def _step_penalty_target_reward(self) -> Tuple[float, bool]:
        """
        Returns a fixed step penalty (1 / max_steps) for each step. Only a goal-region-reaching step
        is rewarded with 1 (and terminates the episode)
        @return:
        """
        d = math.dist([self.x, self.y], [0.0, 0.0])
        d_o = min(abs(self.theta), 2 * math.pi - abs(self.theta))
        diff_lin_vel = abs(self.lin_vel_target - self.lin_vel)
        diff_phi = abs(self.phi_target - self.phi)

        if d < self.goal_tolerance_pos \
                and d_o < self.goal_tolerance_theta \
                and diff_lin_vel < self.goal_tolerance_lin_vel \
                and diff_phi < self.goal_tolerance_phi:  # target reached
            return 1, True

        max_steps_reached = self.action_count >= self.max_steps - 1
        reward = - (1 / self.max_steps)  # step penalty
        return reward, max_steps_reached

    def _observations(self) -> Tuple[float, float, float, float, float, float, float]:
        """
        Returns the system's state (+ targets) in the observation space
        @return:(all floats) x_obs, y_obs, theta_obs, lin_vel_obs, phi_obs, lin_vel_target_obs, phi_target_obs
        """
        x_obs = self.x
        y_obs = self.y
        theta_obs = map_to_diff_range(0, np.pi * 2, self.observation_space_low, self.observation_space_high, self.theta)

        # design choice: the mapping from lin_vel to lin_vel_obs is done with a slightly bigger range to also encompass
        # boundary violations into the [-1, 1] observation space
        lin_vel_obs = map_to_diff_range(self.lin_vel_obs_min, self.lin_vel_obs_max,
                                        self.observation_space_low, self.observation_space_high, self.lin_vel)

        phi_obs = map_to_diff_range(self.dynamics.phi_min, self.dynamics.phi_max,
                                    self.observation_space_low, self.observation_space_high, self.phi)

        return x_obs, y_obs, theta_obs, lin_vel_obs, phi_obs, self.lin_vel_target_obs, self.phi_target_obs

    def get_state(self) -> Tuple[float, float, float, float, float, float, float]:
        """
        returns the current state of the environment
        @return (x, y, theta, lin_vel, phi, lin_vel_target, phi_target)
        """
        return self.x, self.y, self.theta, self.lin_vel, self.phi, self.lin_vel_target, self.phi_target

    def step(self, action: List[float]):
        """
        Transition function of the environment
        @param action: the action that should be enacted in the environment. (in the normalized range)
        @return: observation (np.Array (x, y, theta) in observation range), reward, terminated, info (empty dict)
        """

        lin_vel_accl, phi_vel = self.dynamics.convert_normalized_action(action)

        self.last_x = self.x
        self.last_y = self.y
        self.last_theta = self.theta

        self.x, self.y, self.theta, self.lin_vel, self.phi \
            = self.dynamics.step([self.x, self.y, self.theta, self.lin_vel, self.phi], [lin_vel_accl, phi_vel])

        (reward, terminated) = self._reward()

        self.action_count += 1

        self.terminated = terminated

        observation = np.array(self._observations(), dtype=np.float32)

        return observation, reward, terminated, {}

    def get_sample_from_training_set_with_added_noise(self) -> Tuple[float, float, float, float, float, float, float]:
        """
        samples a config from the training_set and adds gaussian noise (mean=0, standard deviation = std_noise)
        @return: x,y,theta,lin_vel,phi,lin_vel_target,phi_target
        """

        [(x, y, theta, lin_vel, phi, lin_vel_target, phi_target)] = random.sample(self.training_set, 1)
        # position
        x = x + random.gauss(0, self.std_noise)
        y = y + random.gauss(0, self.std_noise)
        # orientation (wrapped with % 2pi)

        theta = (theta + random.gauss(0, self.std_noise)) % (2 * np.pi)

        # linear velocity. clipped
        lin_vel = lin_vel + random.gauss(0, self.std_noise)
        lin_vel_target = lin_vel_target + random.gauss(0, self.std_noise)

        lin_vel, lin_vel_target = np.clip([lin_vel, lin_vel_target], self.dynamics.lin_vel_min,
                                          self.dynamics.lin_vel_max)

        # phi (steering angle). clipped
        phi_target = phi_target + random.gauss(0, self.std_noise)
        phi = phi + random.gauss(0, self.std_noise)
        phi, phi_target = np.clip([phi, phi_target],
                                  self.dynamics.phi_min, self.dynamics.phi_max)
        return x, y, theta, lin_vel, phi, lin_vel_target, phi_target

    def sample_state(self) -> Tuple[float, float, float, float, float,float,float]:
        """
        samples a new state:
        x,y : sampled (via polar coordinates) from the circle around (0,0) with r in [reset_r_min, reset_r_max]
            theta: uniformly sampled from [0, 2pi]
        lin_vel: uniformly sampled from [dynamics.lin_vel_min, dynamics.lin_vel_max]
        phi: uniformly sampled from [dynamics.phi_min, dynamics.phi_max]
        lin_vel_target: uniformly sampled from [dynamics.lin_vel_min, dynamics.lin_vel_max]
        phi_target: uniformly sampled from [dynamics.phi_min, dynamics.phi_max]
        """

        r = (self.reset_r_max - self.reset_r_min) * np.random.random() + self.reset_r_min
        phi = 2 * math.pi * random.random()

        x = math.cos(phi) * r
        y = math.sin(phi) * r

        theta = 2 * math.pi * random.random()

        phi = (self.dynamics.phi_max - self.dynamics.phi_min) * random.random() + self.dynamics.phi_min
        lin_vel = (self.dynamics.lin_vel_max - self.dynamics.lin_vel_min) * random.random() + self.dynamics.lin_vel_min

        phi_target = (self.dynamics.phi_max - self.dynamics.phi_min) * random.random() + self.dynamics.phi_min
        lin_vel_target = (self.dynamics.lin_vel_max - self.dynamics.lin_vel_min) * random.random() \
                         + self.dynamics.lin_vel_min

        return x, y, theta, lin_vel, phi, lin_vel_target, phi_target

    def reset(self) -> ndarray:
        """
        Resets the environment
        if easy_training is set to True the reset state is sampled from the set specified
        by the class param training_set. (noise is added with gauss (mean=0 and standard-deviation=std_noise))

        otherwise, the new state is sampled in the following way:
        x,y : sampled (via polar coordinates) from the circle around (0,0) with r in [reset_r_min, reset_r_max]
        theta: uniformly sampled from [0, 2pi]
        lin_vel: uniformly sampled from [dynamics.lin_vel_min, dynamics.lin_vel_max]
        phi: uniformly sampled from [dynamics.phi_min, dynamics.phi_max]
        lin_vel_target: uniformly sampled from [dynamics.lin_vel_min, dynamics.lin_vel_max]
        phi_target: uniformly sampled from [dynamics.phi_min, dynamics.phi_max]
        @return: the new state in the observation space
        """
        if self.easy_training:
            x, y, theta, lin_vel, phi, lin_vel_target, phi_target \
                = self.get_sample_from_training_set_with_added_noise()

        else:
            x, y, theta, lin_vel, phi, lin_vel_target, phi_target = self.sample_state()

        self.x = x
        self.y = y
        self.theta = theta
        self.lin_vel = lin_vel
        self.phi = phi
        self.lin_vel_target = lin_vel_target
        self.phi_target = phi_target

        self.last_x = self.x
        self.last_y = self.y

        self.phi_target_obs = map_to_diff_range(self.dynamics.phi_min, self.dynamics.phi_max,
                                                self.observation_space_low, self.observation_space_high, self.phi_target)

        self.lin_vel_target_obs = map_to_diff_range(self.dynamics.lin_vel_min, self.dynamics.lin_vel_max,
                                                    self.observation_space_low, self.observation_space_high,
                                                    self.lin_vel_target)

        self.action_count = 0

        return np.array(self._observations(), dtype=np.float32)  # reward, done, info can't be included

    def set_state(self, x_start, y_start, theta_start, lin_vel, phi, lin_vel_target, phi_target, reset_action_count=True):
        """
        sets the system's state according to the input
        @param x_start
        @param y_start
        @param theta_start
        @param lin_vel
        @param phi
        @param lin_vel_target
        @param phi_target
        @return: new system state in the oberservation space
        """
        self.x = x_start
        self.y = y_start
        self.theta = theta_start
        self.lin_vel = lin_vel
        self.phi = phi

        self.phi_target = phi_target
        self.lin_vel_target = lin_vel_target

        self.phi_target_obs = map_to_diff_range(self.dynamics.phi_min, self.dynamics.phi_max,
                                                self.observation_space_low, self.observation_space_high,
                                                self.phi_target)

        self.lin_vel_target_obs = map_to_diff_range(self.dynamics.lin_vel_min, self.dynamics.lin_vel_max,
                                                    self.observation_space_low, self.observation_space_high,
                                                    self.lin_vel_target)

        self.last_x = x_start
        self.last_y = y_start
        self.last_theta = theta_start

        if reset_action_count:
            self.action_count = 0

        observation = np.array(self._observations(), dtype=np.float32)

        self.action_count = 0

        return observation

    def render(self, mode="human", perfect_path=None, overlay_traces=False, screenshot_dir=None) -> Optional[ndarray]:
        if self.artist is None:
            self.artist = AckermannCarArtist(
                car_length=self.dynamics.car_length,
                render_mode="human",
                render_fps=self.metadata['render_fps'],
                screen_width=500,
                screen_height=500,
                trace_mode=self.trace_mode,
                x_max=1
            )
        is_first_state = True if self.action_count == 0 else False
        is_last_state = True if self.terminated else False
        return self.artist.render_environment_state(self.x, self.y, self.theta, self.goal_tolerance_pos,
                                             self.goal_tolerance_theta, perfect_path, overlay_traces,
                                             screenshot_dir, is_first_state, is_last_state, self.last_x, self.last_y)



    def close(self):
        print(f"closing now")
