import math
from typing import List, Tuple

import numpy as np

from dynamics.Dynamics import Dynamics
from motion_planning.maps.map_2d import Map2D
from spaces.State import StateAckermannCarFirstOrder, State
from spaces.Action import ActionAckermannCarFirstOrder
from project_utils import utils


class AckermannCarDynamicsFirstOrder(Dynamics):
    """
    Class for modelling of the (first order) dynamics of an ackermann steering car
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)

        # linear velocity min/max
        self.lin_vel_max: float = self.get_param_from_config("lin_vel_max")
        self.lin_vel_min: float = self.get_param_from_config("lin_vel_min")

        # steering angle min/max
        self.phi_min: float = self.get_param_from_config("phi_min")
        self.phi_max: float = self.get_param_from_config("phi_max")

        self.car_length: float = self.get_param_from_config("L")  # length of the car
        self.delta_t: float = self.get_param_from_config("delta_t")  # time step

    def get_param_string(self):
        return f"""
        lin_vel: ({self.lin_vel_min},{self.lin_vel_max})
        phi: ({self.phi_min},{self.phi_max})
        length: {self.car_length}
        delta_t: {self.delta_t}
        """

    def convert_normalized_action(self, action: List[float], action_space_low=-1, action_space_high=1):
        """
        converts the normalized action into linear velocity and steering angle
        @param action: normalized actions
        @param action_space_low: min value of the action space
        @param action_space_high: max value of the action space
        @return:
        """
        lin_vel = utils.map_to_diff_range(action_space_low, action_space_high,
                                          self.lin_vel_min, self.lin_vel_max, action[0])
        phi = utils.map_to_diff_range(action_space_low, action_space_high,
                                      self.phi_min, self.phi_max, action[1])
        return lin_vel, phi

    def step(self, state: List[float], action: List[float]):
        """
        Transition function for one timestep from the given state to the next based on the given action input
        @param state: current state (x,y,theta)
        @param action: action input (lin_vel, phi)
        @return: next state (x,y,theta)
        """
        assert len(state) == 3, "the initial state consist of three entries (x, y, theta)"
        x_0, y_0, theta_0 = state
        assert len(action) == 2, "this dynamics model only has two control inputs (linear velocity and steering angle)"
        lin_vel, phi = action
        assert self.lin_vel_min <= lin_vel <= self.lin_vel_max, "lin_vel action out of bounds"
        assert self.phi_min <= phi <= self.phi_max, "phi action out of bounds"

        x_1 = x_0 + math.cos(theta_0) * lin_vel * self.delta_t
        y_1 = y_0 + math.sin(theta_0) * lin_vel * self.delta_t
        theta_1 = (theta_0 + (lin_vel / self.car_length) * math.tan(phi) * self.delta_t) % (math.pi * 2)

        return x_1, y_1, theta_1

    def propagate_constant_action(self, init_state: StateAckermannCarFirstOrder, action: ActionAckermannCarFirstOrder,
                                  duration: float) -> Tuple[
                                  List[ActionAckermannCarFirstOrder], List[StateAckermannCarFirstOrder]]:
        """
        Propagates the initial state for (duration / delta_t) steps with constant input
        @param init_state: initial state of the system
        @param action: constant control
        @param duration: duration for which to apply the control
        @return: actions, states
        """
        states = []
        _state = init_state
        n = int(duration / self.delta_t)
        x_0, y_0, theta_0 = init_state.x, init_state.y, init_state.theta
        for _ in range(n):
            (x_1, y_1, theta_1) = self.step([x_0, y_0, theta_0],
                                            [action.lin_vel, action.phi])

            states.append(StateAckermannCarFirstOrder(x_1, y_1, theta_1))
            (x_0, y_0, theta_0) = x_1, y_1, theta_1

        return [action for _ in range(n)], states

    def sample_control(self):
        """
        Samples valid controls for the dynamic-configuration
        @return: linear velocity, steering angle
        """
        lin_vel = np.random.uniform(self.lin_vel_min, self.lin_vel_max)
        phi = np.random.uniform(self.phi_min, self.phi_max)
        return lin_vel, phi

    def sample_action(self) -> ActionAckermannCarFirstOrder:
        lin_vel, phi = self.sample_control()
        return ActionAckermannCarFirstOrder(lin_vel, phi)


    def sample_valid_state(self, scenario_map: Map2D) -> StateAckermannCarFirstOrder:
        x, y = scenario_map.sample_pos_uniform()
        theta = np.random.uniform(0, 2 * np.pi)
        return StateAckermannCarFirstOrder(x, y, theta)
