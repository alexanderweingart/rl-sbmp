from typing import Tuple

import onnx
import onnxruntime as ort
import numpy as np
from numpy import ndarray

from rl.environments.car import CustomEnv
from typing import List


class OnnxModelWrapper:
    """
    Helper class that loads a model from an onnx file and provides a predict API matching the
    PPO model from stablebaselines3
    """

    def __init__(self, env: CustomEnv, onnx_path: str):
        """
        @param env: The model's environment
        @param onnx_path: The path to the onnx file
        """
        self.onnx_path = onnx_path
        self.model = onnx.load(self.onnx_path)
        onnx.checker.check_model(self.model)
        self.ort_sess = ort.InferenceSession(self.onnx_path)
        self.env = env

    def predict(self, observation: List[float], deterministic: bool) -> Tuple[ndarray, List]:
        """
        Predicts the best action for the given input.
        @param observation: List of observations in float
        @param deterministic: THIS VALUE IS IGNORED. ONLY EXISTS FOR EASY INTEGRATION.
        @return: actions, []  (NO STATES WILL BE RETURNED)
        """
        observation = np.array([observation])
        action, value = self.ort_sess.run(None, {"input": observation})

        [action_clipped] = np.clip(action, self.env.action_space_low, self.env.action_space_high)
        return action_clipped, []
