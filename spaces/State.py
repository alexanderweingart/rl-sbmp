from __future__ import annotations
import numpy as np
import math
from typing import Tuple

from numpy import ndarray
import math


class State:
    """
    abstract class
    the concrete implementation is defined a part of the dynamics
    """

    def vector(self) -> ndarray:
        raise NotImplementedError

    def vector_weighted(self) -> ndarray:
        raise NotImplementedError

    def distance_to(self, s2: State) -> float:
        raise NotImplementedError

    def get_pose_tuple(self) -> Tuple[float, ...]:
        raise NotImplementedError

    def distance_to_split_components(self, s2: State):
        raise NotImplementedError


class StateAckermannCarFirstOrder(State):
    WEIGHT_D = 1  # weight of the position distance
    WEIGHT_DO = 1  # weight of the orientation distance

    def __init__(self, x: float, y: float, theta: float):
        self.x = x
        self.y = y
        while theta < 0:
            theta += 2 * np.pi
        self._theta = theta
        self.change_counter_theta = 1

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        self._theta = value
        self.change_counter_theta += 1
        if self.change_counter_theta > 2:
            print("wtf")

    def position_distance_to(self, s2: StateAckermannCarFirstOrder) -> float:
        """
        Returns the distance (regarding the position) to the given state
        @param s2: other state
        @return: distance
        """
        return math.dist((self.x, self.y), (s2.x, s2.y))

    def orientation_distance_to(self, s2: StateAckermannCarFirstOrder) -> float:
        """
        Returns the distance (regarding the orientation) to the given state
        @param s2: other state
        @return: distance
        """
        o_abs_diff = np.abs(s2.theta - self.theta)
        return min([o_abs_diff, 2 * np.pi - o_abs_diff])

    # def distance_to(self, s2: StateAckermannCarFirstOrder, w_d=WEIGHT_D, w_do=WEIGHT_DO) -> float:
    #     d = self.position_distance_to(s2)
    #     d_o = self.orientation_distance_to(s2)
    #     return d * w_d + w_do * d_o
    def distance_to(self, s2: State) -> float:
        vec1 = self.vector_weighted()
        vec2 = s2.vector_weighted()
        return math.dist(vec1, vec2)

    def distance_to_split_components(self, s2: State):
        assert type(s2) == StateAckermannCarFirstOrder
        s2: StateAckermannCarFirstOrder
        d = self.position_distance_to(s2)
        d_o = self.orientation_distance_to(s2)
        return d, d_o


    def vector(self) -> ndarray:
        return np.array([self.x, self.y, self.theta], dtype=float)

    def vector_weighted(self, w_d=WEIGHT_D, w_do=WEIGHT_DO) -> ndarray:
        theta_x = np.cos(self.theta)
        theta_y = np.sin(self.theta)
        return np.array([self.x * w_d, self.y * w_d, theta_x * w_do, theta_y * w_do])


class CostStateAckermannCarFirstOrder(StateAckermannCarFirstOrder):
    def __init__(self, x, y, theta, cost_to_come):
        super().__init__(x, y, theta)
        self.cost_to_come = cost_to_come
        self.WEIGHT_C = 0.001

    def vector_weighted(self, w_d=None, w_do=None, w_c=None) -> ndarray:
        if w_d is None:
            w_d = self.WEIGHT_D
        if w_do is None:
            w_do = self.WEIGHT_DO
        if w_c is None:
            w_c = self.WEIGHT_C
        theta_x = np.cos(self.theta)
        theta_y = np.sin(self.theta)
        return np.array([self.x * w_d, self.y * w_d, theta_x * w_do, theta_y * w_do, self.cost_to_come * w_c])


class StateAckermannCarSecondOrder(StateAckermannCarFirstOrder):
    WEIGHT_LIN_VEL = 1
    WEIGHT_PHI = 1

    def __init__(self, x, y, theta, lin_vel, phi):
        super().__init__(x, y, theta)
        self.lin_vel = lin_vel
        self.phi = phi

    def lin_vel_distance_to(self, s2: StateAckermannCarSecondOrder):
        return abs(s2.lin_vel - self.lin_vel)

    def phi_distance_to(self, s2: StateAckermannCarSecondOrder):
        return abs(s2.phi - self.phi)

    def distance_to(self, s2: StateAckermannCarSecondOrder, w_d=None, w_do=None, w_lin_vel=None, w_phi=None) -> float:
        if w_d is None:
            w_d = self.WEIGHT_D
        if w_do is None:
            w_do = self.WEIGHT_DO
        if w_lin_vel is None:
            w_lin_vel = self.WEIGHT_LIN_VEL
        if w_phi is None:
            w_phi = self.WEIGHT_PHI

        d = self.position_distance_to(s2)
        d_o = self.orientation_distance_to(s2)
        d_lin_vel = self.lin_vel_distance_to(s2)
        d_phi = self.phi_distance_to(s2)
        return float(d * w_d + w_do * d_o + d_lin_vel * w_lin_vel + d_phi * w_phi)

    def vector(self) -> ndarray:
        return np.array([self.x, self.y, self.theta, self.lin_vel, self.phi], dtype=float)

    def vector_weighted(self, w_d=None, w_do=None, w_lin_vel=None, w_phi=None) -> ndarray:
        if w_d is None:
            w_d = self.WEIGHT_D
        if w_do is None:
            w_do = self.WEIGHT_DO
        if w_lin_vel is None:
            w_lin_vel = self.WEIGHT_LIN_VEL
        if w_phi is None:
            w_phi = self.WEIGHT_PHI

        return np.array([self.x * w_d, self.y * w_d, self.theta * w_do, self.lin_vel * w_lin_vel, self.phi * w_phi],
                        dtype=float)

    def distance_to_split_components(self, s2: State):
        assert type(s2) == StateAckermannCarSecondOrder
        s2: StateAckermannCarSecondOrder
        d = self.position_distance_to(s2)
        d_o = self.orientation_distance_to(s2)
        d_lin_vel = self.lin_vel_distance_to(s2)
        d_phi = self.phi_distance_to(s2)
        return d, d_o, d_lin_vel, d_phi


class CostStateAckermannCarSecondOrder(StateAckermannCarSecondOrder):
    def __init__(self, x, y, theta, lin_vel, phi, cost_to_come):
        super().__init__(x, y, theta, lin_vel, phi)
        self.cost_to_come = cost_to_come
        # 1/1000 to keep this portion of the distance in the same ballpark as the others
        self.WEIGHT_COST_TO_COME = 0.001

    def vector_weighted(self, w_d=None, w_do=None, w_lin_vel=None, w_phi=None, w_c=None) -> ndarray:
        if w_d is None:
            w_d = self.WEIGHT_D
        if w_do is None:
            w_do = self.WEIGHT_DO
        if w_lin_vel is None:
            w_lin_vel = self.WEIGHT_LIN_VEL
        if w_phi is None:
            w_phi = self.WEIGHT_PHI
        if w_c is None:
            w_c = self.WEIGHT_COST_TO_COME
        return np.array([self.x * w_d, self.y * w_d, self.theta * w_do, self.lin_vel * w_lin_vel,
                         self.phi * w_phi, self.cost_to_come * w_c],
                        dtype=float)


