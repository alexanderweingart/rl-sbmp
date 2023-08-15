from enum import Enum
from stable_baselines3.common.base_class import BaseAlgorithm
from dynamics.Dynamics import Dynamics
from spaces.State import State
from spaces.Action import Action
from typing import Tuple
from rl.environments.extended_env import ExtendedEnv
from rl.environments.car import CustomEnv

import gym
from typing import List


class ExtensionModes(Enum):
    RL = "RL"
    MCP = "MCP"
    GUIDED_MCP = "GUIDED_MCP"
    RL_ONNX = "RL_ONNX"


class Driver:
    def propagate_rl(self, q_select: State, q_rand: State, radius=0.5, scale_theta_diff=False,
                     go_straight_mode=False,
                     deterministic_policy=True,
                     use_onnx_model: bool = False, add_cost: bool = False)\
            -> Tuple[List[Action], List[State]]:
        raise NotImplementedError

    def propagate_mcp(self, s_init: State, t_sec_min: float, t_sec_max: float, add_cost: bool = False) \
            -> Tuple[List[Action], List[State]]:
        """
        Monte-Carlo Propagation from s_init
        @param add_cost: controls if StateSpace is augmented with cost dimension
        @param s_init: initial state
        @param t_sec_min: minimum time the controls are applied (in sec)
        @param t_sec_max: maximum time the controls are applied (in sec)
        @return: actions, states, duration
        """
        raise NotImplementedError

    def propagate_mcp_guided(self, s_init: State, s_target: State, k: int, t_sec_min: float, t_sec_max: float,
                             add_cost: bool = False) \
            -> Tuple[List[Action], List[State]]:
        """
        Guided monte carlo propagation from s_init towards s_target
        @param add_cost: [use this for AO-RRT] adds the cost-to-come to each state on the trajectory
        @param t_sec_max:
        @param t_sec_min:
        @param s_init: initial state
        @param s_target: target state
        @param k: number of sampled trajectories
        @return: trajectory which finishes closest to s_target (actions, states, duration)
        """
        raise NotImplementedError
