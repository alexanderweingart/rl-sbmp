import logging
from typing import List
from abc import abstractmethod
import yaml
from motion_planning.maps.map_2d import Map2D
from spaces.State import State
from typing import Tuple
from spaces.Action import Action

class Dynamics:
    delta_t: float
    def __init__(self, config_path: str):
        self.logger = logging.getLogger(__file__)

        if config_path is None:
            self.logger.warning("the config_path is None")
            raise ValueError

        self.config = yaml.safe_load(open(config_path))

    def get_param_from_config(self, key: str):
        """
        Extract parameter from the configuration dict

        @raise Value Error if the key cannot be found
        @param key: key the parameter is stored under in the config yaml
        @return: the parameter's value
        """
        if key not in self.config:
            self.logger.error(f"key {key} is could not be found in the configuration!")
            raise ValueError
        return self.config[key]

    def get_param_string(self) -> str:
        raise NotImplementedError

    def convert_normalized_action(self, action: List[float], action_space_low: float, action_space_high: float):
        raise NotImplementedError

    def step(self, state: List[float], action: List[float]):
        raise NotImplementedError

    def sample_valid_state(self, scenario_map: Map2D) -> State:
        raise NotImplementedError

    def sample_control(self) -> Tuple[float, float]:
        raise NotImplementedError

    def sample_action(self) -> Action:
        raise NotImplementedError

    def propagate_constant_action(self, s_init: State, action: Action, t: float):
        raise NotImplementedError



