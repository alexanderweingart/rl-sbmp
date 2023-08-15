from numpy import ndarray
import numpy as np


class Action:
    """
    abstract class
    the concrete implementation is defined as part of the Dynamics
    """

    def vector(self) -> ndarray:
        raise NotImplementedError


class ActionAckermannCarFirstOrder(Action):
    lin_vel: float
    phi: float

    def __init__(self, lin_vel: float, phi: float):
        self.lin_vel = lin_vel
        self.phi = phi

    def vector(self) -> ndarray:
        return np.array([self.lin_vel, self.phi])


class ActionAckermannCarSecondOrder(Action):
    lin_accl: float
    phi_vel: float

    def __init__(self, lin_accl: float, phi_vel: float):
        self.lin_accl = lin_accl
        self.phi_vel = phi_vel

    def vector(self) -> ndarray:
        return np.array([self.lin_accl, self.phi_vel])

