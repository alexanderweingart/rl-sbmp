import logging
from typing import Tuple, List
import math
from dynamics.Dynamics import Dynamics
import numpy as np
from project_utils import utils
from motion_planning.maps.map_2d import Map2D
from spaces.State import StateAckermannCarSecondOrder
from spaces.Action import ActionAckermannCarSecondOrder


class AckermannCarDynamicsSecondOrder(Dynamics):
    """
    Class for modelling of the (second order) dynamics of an ackermann steering car
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)

        # linear velocity min/max
        self.lin_vel_max: float = self.get_param_from_config("lin_vel_max")
        self.lin_vel_min: float = self.get_param_from_config("lin_vel_min")

        # linear acceleration min/max
        self.lin_accl_min: float = self.get_param_from_config("lin_accl_min")
        self.lin_accl_max: float = self.get_param_from_config("lin_accl_max")

        # steering angle min/max
        self.phi_min: float = self.get_param_from_config("phi_min")
        self.phi_max: float = self.get_param_from_config("phi_max")

        # angular velocity of the steering angle min/max
        self.phi_vel_min: float = self.get_param_from_config("phi_vel_min")
        self.phi_vel_max: float = self.get_param_from_config("phi_vel_max")

        self.car_length: float = self.get_param_from_config("L")
        self.delta_t: float = self.get_param_from_config("delta_t")  # time step

        self.clip_lin_vel: bool = self.get_param_from_config("clip_lin_vel")
        if not self.clip_lin_vel:
            raise NotImplementedError("In the current version of the Dynamics linear velocity is always clipped!")

    def get_param_string(self):
        return f"""
        lin_vel:  ({self.lin_vel_min}, {self.lin_accl_max})
        lin_accl: ({self.lin_accl_min}, {self.lin_accl_max})
        phi: ({self.phi_min}, {self.phi_max})
        car_length: {self.car_length}
        delta_t: {self.delta_t}
        """

    def step(self, state: List[float], action: List[float]):
        """
        Transition function for one timestep from the given state to the next based on the given action input
        @param state: current state (x, y, theta, linear velocity, steering angle)
        @param action: action input (lin_vel, phi)
        @return: next state (x, y, theta, linear velocity, steering angle)
        """

        assert len(
            state) == 5, f"the initial state consist of five entries (x, y, theta, lin_vel, phi) [is: {len(state)}]"
        x_0, y_0, theta_0, lin_vel_0, phi_0 = state

        assert len(action) == 2, "this dynamics model only has two control inputs (linear acceleration and angular " \
                                 "velocity of the steering angle)"
        lin_accl, phi_vel = action

        x_1 = x_0 + math.cos(theta_0) * lin_vel_0 * self.delta_t
        y_1 = y_0 + math.sin(theta_0) * lin_vel_0 * self.delta_t
        theta_1 = (theta_0 + (lin_vel_0 / self.car_length) * math.tan(phi_0) * self.delta_t) % (math.pi * 2)

        lin_vel_1 = lin_vel_0 + lin_accl * self.delta_t
        lin_vel_1 = np.clip(lin_vel_1, self.lin_vel_min, self.lin_vel_max)

        phi_1 = phi_0 + phi_vel * self.delta_t
        phi_1 = np.clip(phi_1, self.phi_min, self.phi_max)

        return x_1, y_1, theta_1, lin_vel_1, phi_1

    def convert_normalized_action(self, action: List[float], action_space_low=-1, action_space_high=1):
        """
        converts the normalized action into linear velocity and steering angle
        @param action: normalized actions
        @param action_space_low: min value of the action space
        @param action_space_high: max value of the action space
        @return:
        """
        lin_accl = utils.map_to_diff_range(action_space_low, action_space_high,
                                           self.lin_accl_min, self.lin_accl_max, action[0])
        phi_vel = utils.map_to_diff_range(action_space_low, action_space_high,
                                          self.phi_vel_min, self.phi_vel_max, action[1])
        return lin_accl, phi_vel

    def sample_valid_state(self, scenario_map: Map2D) -> StateAckermannCarSecondOrder:
        x, y = scenario_map.sample_pos_uniform()
        lin_vel = np.random.uniform(self.lin_vel_min, self.lin_vel_max)
        phi = np.random.uniform(self.phi_min, self.phi_max)
        theta = np.random.uniform(0, 2 * np.pi)
        return StateAckermannCarSecondOrder(x, y, theta, lin_vel, phi)

    def sample_control(self) -> Tuple[float, float]:
        lin_accl = np.random.uniform(self.lin_accl_min, self.lin_accl_max)
        phi_vel = np.random.uniform(self.phi_vel_min, self.phi_vel_max)
        return lin_accl, phi_vel

    def sample_action(self) -> ActionAckermannCarSecondOrder:
        lin_accl, phi_vel = self.sample_control()
        return ActionAckermannCarSecondOrder(lin_accl, phi_vel)

    def propagate_constant_action(self, init_state: StateAckermannCarSecondOrder, action: ActionAckermannCarSecondOrder,
                                  duration: float) -> Tuple[
        List[ActionAckermannCarSecondOrder], List[StateAckermannCarSecondOrder]]:
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
        x_0, y_0, theta_0, lin_vel_0, phi_0 = init_state.x, init_state.y, init_state.theta, init_state.lin_vel, init_state.phi
        for _ in range(n):
            (x_1, y_1, theta_1, lin_vel_1, phi_1) = self.step([x_0, y_0, theta_0, lin_vel_0, phi_0],
                                                              [action.lin_accl, action.phi_vel])

            states.append(StateAckermannCarSecondOrder(x_1, y_1, theta_1, lin_vel_1, phi_1))
            (x_0, y_0, theta_0, lin_vel_0, phi_0) = x_1, y_1, theta_1, lin_vel_1, phi_1

        return [action for _ in range(n)], states
