import os.path

import pytest
from spaces.State import CostStateAckermannCarFirstOrder as State
from driver.ackermann_driver import AckermannCarFirstOrderDriver
from driver.ackermann_driver import get_intermediate_target
from dynamics.AckermannCarDynamicsFirstOrder import AckermannCarDynamicsFirstOrder
import numpy as np


class TestAckermannDriver:
    config_path = os.path.join(os.path.dirname(__file__), "star_models",
                               "ackermann", "vel_ctrl", "config.yaml")
    model_path = os.path.join(os.path.dirname(__file__), "star_models",
                              "ackermann", "vel_ctrl", "ppo_1678273546.zip")
    onnx_model_path = os.path.join(os.path.dirname(__file__), "star_models",
                                   "ackermann", "vel_ctrl", "ppo_1678273546.onnx")

    @pytest.mark.parametrize("s_init, s_target, step_size, expected_state", [
        (
            State(0, 0, 0, 0),  # straight line x
            State(5, 0, 0, 0),
            1,
            (1, 0, 0)
        ),
        (
            State(0, 0, 0.5 * np.pi, 0),  # straight line y
            State(0, 5, 0.5 * np.pi, 0),
            1,
            (0, 1, 0.5 * np.pi)
        ),
        (
            State(0, 0, 0.25 * np.pi, 0),  # wrap around diff (target > init)
            State(0, 5, 1.75 * np.pi, 0),
            1,
            (0, 1, 0.15 * np.pi)
        ),
        (
            State(0, 0, 1.75 * np.pi, 0),  # wrap around diff (init > target)
            State(0, 5, 0.25 * np.pi, 0),
            1,
            (0, 1, 1.85 * np.pi)
        ),
    ])
    def test_get_intermediate_target(self, s_init: State, s_target: State, step_size: float, expected_state: State):
        first_state = get_intermediate_target(s_init=s_init, s_target=s_target, step_size=step_size, scale_theta=True)
        assert first_state == expected_state
