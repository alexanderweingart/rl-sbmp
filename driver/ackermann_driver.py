import numpy as np
import copy
import math
from project_utils import transformations as tf
from spaces.Action import ActionAckermannCarFirstOrder
from spaces.Action import ActionAckermannCarSecondOrder

from dynamics.AckermannCarDynamicsFirstOrder import AckermannCarDynamicsFirstOrder
from dynamics.AckermannCarDynamicsSecondOrder import AckermannCarDynamicsSecondOrder
from spaces.State import CostStateAckermannCarFirstOrder
from spaces.State import CostStateAckermannCarSecondOrder
from spaces.State import StateAckermannCarFirstOrder
from spaces.State import StateAckermannCarSecondOrder
from driver.driver import Driver
from typing import Tuple, Optional, List
from rl.environments.car import CustomEnv as AckermannCarEnvFirstOrder
from rl.environments.accl_car import CustomEnv as AckermannCarEnvSecondOrder

from stable_baselines3 import PPO
from rl.onnx_helper.onnx_wrapper import OnnxModelWrapper

def transform(pose, target_pose):
    (x, y, theta) = pose
    (x_t, y_t, theta_t) = target_pose
    return tf.transform_to_target_frame(x, y, theta, x_t, y_t, theta_t)


def retransform(pose, target_pose):
    (x, y, theta) = pose
    (x_t, y_t, theta_t) = target_pose
    return tf.transform_from_target_frame(x, y, theta, x_t, y_t, theta_t)


class AckermannCarFirstOrderDriver(Driver):
    def __init__(self, config_path: str, model_path: str,
                 dynamics: AckermannCarDynamicsFirstOrder, model_path_onnx: Optional[str] = None):

        self.env = AckermannCarEnvFirstOrder(config_path)

        if model_path is not None:
            self.model = PPO.load(model_path, self.env)
        else:
            self.model = None
        self.dynamics = dynamics
        if model_path_onnx is not None:
            self.model_onnx = OnnxModelWrapper(self.env, model_path_onnx)
        else:
            self.model_onnx = None

    def propagate_rl(self, q_select: StateAckermannCarFirstOrder, q_rand: StateAckermannCarFirstOrder, radius=0.5, scale_theta_diff=True,
                     go_straight_mode=False, deterministic_policy=True, use_onnx_model=False, add_cost: bool = False,
                     sliding_window_mode=True, sliding_window_max_iter: int = 40) \
            -> Tuple[List[ActionAckermannCarFirstOrder], List[StateAckermannCarFirstOrder]]:
        if sliding_window_mode:
            assert not go_straight_mode, "sliding window mode flag, has higher priority and " \
                                         "is not compatible with go_straight_mode "
            return self._propagate_rl_sliding_window(q_select, q_rand, radius, deterministic_policy,
                                                     add_cost, sliding_window_max_iter)
        else:
            return self._propagate_rl(q_select, q_rand, radius, scale_theta_diff,
                                      go_straight_mode, deterministic_policy, use_onnx_model, add_cost)

    def _propagate_rl(self, q_select: StateAckermannCarFirstOrder, q_rand: StateAckermannCarFirstOrder, radius=0.5, scale_theta_diff=True,
                      go_straight_mode=False, deterministic_policy=True, use_onnx_model=False, add_cost: bool = False) \
            -> Tuple[List[ActionAckermannCarFirstOrder], List[StateAckermannCarFirstOrder]]:
        """
        Propagate the AckermanCar System from q_select towards q_rand using the RL-model
        @param q_select: initial state
        @param q_rand: target state
        @param radius: maximum extension radius
        @param scale_theta_diff: (bool) if activated,  theta is scaled according to the ratio between extension-radius
        and the original translational difference between q_rand and q_select
        @param go_straight_mode: (bool) if activated, theta will be set to the directional vector from q_select to q_rand
                                        (if the distance is > the extension radius. inside the critical area around this
                                        option is ignored)
        @param deterministic_policy: (bool) flag for the rl model prediction
        @param use_onnx_model: (bool) flag. if this is set to True, the model will be loaded inside the ONNX runtime
        @param add_cost: (bool) flag. if this is set to True, the returned states will be CostStates and include the cost_to_come
                                if this is activated, q_select needs to be a CostState as well
        @return: list of Actions, list of States
        """

        _q_rand = copy.copy(q_rand)

        x_start = _q_rand.x - q_select.x
        y_start = _q_rand.y - q_select.y

        r = math.dist([x_start, y_start], [0, 0])
        if r > radius:  # the model is not ready for bigger distances --> max step = 0.5
            x_start = (x_start / r) * radius
            y_start = (y_start / r) * radius

        if go_straight_mode and (r > radius):  # only activate the go-straight-mode in proximity to the goal
            new_theta = math.atan2(y_start, x_start)
            while new_theta < 0:
                new_theta += 2 * np.pi
            _q_rand.theta = new_theta

        if scale_theta_diff:
            scale = min(radius / r, 1)
            theta_diff = _q_rand.theta - q_select.theta
            direction = 1 if theta_diff > 0 else -1
            if (math.pi * 2 - abs(theta_diff)) < abs(theta_diff):
                theta_diff = (direction * -1) * (math.pi * 2 - abs(theta_diff))
            theta_target = (q_select.theta +
                            (theta_diff * scale)) % (2 * math.pi)
        else:
            theta_target = _q_rand.theta

        adjusted_target_x = q_select.x + x_start
        adjusted_target_y = q_select.y + y_start
        (x_start, y_start, theta_start) = transform([q_select.x, q_select.y, q_select.theta], [
            adjusted_target_x, adjusted_target_y, theta_target])

        obs = self.env.set_state(x_start, y_start, theta_start)

        done = False
        states = []
        actions = []
        while not done:
            if use_onnx_model:
                action, _states = self.model_onnx.predict(
                    obs, deterministic=deterministic_policy)
            else:
                action, _states = self.model.predict(
                    obs, deterministic=deterministic_policy)

            obs, rewards, done, info = self.env.step(action)
            state = self.env.get_state()
            x_state = state[0]
            y_state = state[1]
            theta_state = state[2]

            (x, y, theta) = retransform([x_state, y_state, theta_state], [
                adjusted_target_x, adjusted_target_y, theta_target])
            q = StateAckermannCarFirstOrder(x, y, theta)
            states.append(q)
            action = self.dynamics.convert_normalized_action(action)
            u = ActionAckermannCarFirstOrder(action[0], action[1])
            actions.append(u)

            if add_cost:
                assert type(q_select) == CostStateAckermannCarFirstOrder
                q_select: CostStateAckermannCarFirstOrder
                c_init = q_select.cost_to_come
                cstates = [CostStateAckermannCarFirstOrder(s.x, s.y, s.theta, c_init + (i + 1)) for (i, s) in enumerate(states)]
                states = cstates

            if done:
                break

        return actions, states


    def _propagate_rl_sliding_window(self, q_select: StateAckermannCarFirstOrder, q_rand: StateAckermannCarFirstOrder, radius: float = 0.5,
                                     deterministic_policy: bool = True, add_cost: bool = False, max_steps: int = 20) \
            -> Tuple[List[ActionAckermannCarFirstOrder], List[StateAckermannCarFirstOrder]]:

        _q_rand = copy.copy(q_rand)

        last_state = q_select

        states = []
        actions = []

        for i in range(max_steps):
            adjusted_target_x, adjusted_target_y, adjusted_target_theta = self.get_intermediate_target(last_state,
                                                                                                       q_rand,
                                                                                                       radius)
            (x_start, y_start, theta_start) = transform([last_state.x, last_state.y, last_state.theta], [
                adjusted_target_x, adjusted_target_y, adjusted_target_theta])

            obs = self.env.set_state(x_start, y_start, theta_start)

            action, _states = self.model_onnx.predict(
                obs, deterministic=deterministic_policy)

            obs, rewards, done, info = self.env.step(action)
            state = self.env.get_state()
            x_state = state[0]
            y_state = state[1]
            theta_state = state[2]

            (x, y, theta) = retransform([x_state, y_state, theta_state], [
                adjusted_target_x, adjusted_target_y, adjusted_target_theta])

            q = StateAckermannCarFirstOrder(x, y, theta)
            states.append(q)
            action = self.dynamics.convert_normalized_action(action)
            u = ActionAckermannCarFirstOrder(action[0], action[1])
            actions.append(u)

            last_state = q

            if done:
                break

        if add_cost:
            assert type(q_select) == CostStateAckermannCarFirstOrder
            q_select: CostStateAckermannCarFirstOrder
            c_init = q_select.cost_to_come
            cstates = [CostStateAckermannCarFirstOrder(s.x, s.y, s.theta, c_init + (i + 1)) for (i, s) in enumerate(states)]
            states = cstates

        return actions, states

    def propagate_mcp(self, s_init: StateAckermannCarFirstOrder, t_sec_min: float, t_sec_max: float, add_cost: bool = False):
        """
        Monte-Carlo Propagation from s_init
        @param s_init: initial state
        @param t_sec_min: minimum time the controls are applied (in sec)
        @param t_sec_max: maximum time the controls are applied (in sec)
        @return: actions, states, duration
        """
        t = np.random.uniform(t_sec_min, t_sec_max)
        action = self.dynamics.sample_action()
        actions, states = self.dynamics.propagate_constant_action(s_init, action, t)
        if add_cost:
            assert type(s_init) == CostStateAckermannCarFirstOrder
            s_init: CostStateAckermannCarFirstOrder
            cost_init = s_init.cost_to_come
            cstates = [CostStateAckermannCarFirstOrder(s.x, s.y, s.theta, cost_init + (i + 1)) for (i, s) in enumerate(states)]
            states = cstates

        return actions, states

    def propagate_mcp_guided(self, s_init: StateAckermannCarFirstOrder, s_target: StateAckermannCarFirstOrder, k: int, t_sec_min: float, t_sec_max: float,
                             add_cost: bool = False):
        """
        Guided monte carlo propagation from s_init towards s_target
        @param t_sec_max:
        @param t_sec_min:
        @param s_init: initial state
        @param s_target: target state
        @param k: number of sampled trajectories
        @return: trajectory which finishes closest to s_target (actions, states, duration)
        """
        min_dist = np.inf
        (best_actions, best_states) = [], []

        for i in range(k):
            _actions, _states = self.propagate_mcp(s_init, t_sec_min, t_sec_max, add_cost=add_cost)
            final_state = _states[-1]
            _dist = final_state.distance_to(s_target)
            if _dist < min_dist:
                (best_actions, best_states) = _actions, _states
                min_dist = _dist

        if add_cost:
            assert type(s_init) == CostStateAckermannCarFirstOrder
            s_init: CostStateAckermannCarFirstOrder
            cost_init = s_init.cost_to_come
            best_states = [CostStateAckermannCarFirstOrder(s.x, s.y, s.theta, cost_init + (i + 1)) for (i, s) in
                       enumerate(best_states)]

        return best_actions, best_states

    @classmethod
    def get_intermediate_target(cls, s_init: StateAckermannCarFirstOrder, s_target: StateAckermannCarFirstOrder,
                                step_size: float, scale_theta: bool = True) -> Tuple[float, float, float]:
        """
        Returns an intermediate target based on linear interpolation and a max step size
        @param scale_theta: (bool) flag. if activated (default) theta is scaled according to the ratio between the
                                         translational distance and the step_size
        @param s_init: initial state
        @param s_target:  target state
        @param step_size: maximum length of the connection step
        @return: intermediate target (x,y,theta)
        """

        diff_pos = s_init.position_distance_to(s_target)

        if diff_pos < step_size:  # target is close enough
            return s_target.x, s_target.y, s_target.theta

        dir_vector = np.asarray((s_target.x - s_init.x, s_target.y - s_init.y)) / diff_pos
        scaled_dir_vector = dir_vector * step_size

        scale = min(step_size / diff_pos, 1)

        if scale_theta:
            theta_diff = s_target.theta - s_init.theta
            direction = 1 if theta_diff > 0 else -1

            if (math.pi * 2 - abs(theta_diff)) < abs(theta_diff):
                theta_diff = (math.pi * 2 - abs(theta_diff))
                direction *= -1

            theta_diff *= scale
            theta_target = (s_init.theta + direction * theta_diff) % (2 * math.pi)
        else:
            theta_target = s_target.theta

        return scaled_dir_vector[0]+s_init.x, scaled_dir_vector[1]+s_init.y, theta_target


class AckermannCarSecondOrderDriver(Driver):
    def __init__(self, config_path: str, model_path: str,
                 dynamics: AckermannCarDynamicsSecondOrder, model_path_onnx: Optional[str] = None):

        self.env = AckermannCarEnvSecondOrder(config_path)

        if model_path is not None:
            self.model = PPO.load(model_path, self.env)
        else:
            self.model = None
        self.dynamics = dynamics
        if model_path_onnx is not None:
            self.model_onnx = OnnxModelWrapper(self.env, model_path_onnx)
        else:
            self.model_onnx = None

    def propagate_rl(self, q_select: StateAckermannCarSecondOrder, q_rand: StateAckermannCarSecondOrder, radius=0.5, scale_theta_diff=True,
                     go_straight_mode=False, deterministic_policy=True, use_onnx_model=False, add_cost: bool = False,
                     sliding_window_mode=True, sliding_window_max_iter: int = 40) \
            -> Tuple[List[ActionAckermannCarSecondOrder], List[StateAckermannCarSecondOrder]]:
        if sliding_window_mode:
            assert not go_straight_mode, "sliding window mode flag, has higher priority and " \
                                         "is not compatible with go_straight_mode "
            return self._propagate_rl_sliding_window(q_select, q_rand, radius, deterministic_policy,
                                                     add_cost, sliding_window_max_iter)
        else:
            return self._propagate_rl(q_select, q_rand, radius, scale_theta_diff,
                                      go_straight_mode, deterministic_policy, use_onnx_model, add_cost)

    def _propagate_rl(self, q_select: StateAckermannCarSecondOrder, q_rand: StateAckermannCarSecondOrder, radius=0.5, scale_theta_diff=True,
                      go_straight_mode=False, deterministic_policy=True, use_onnx_model=False, add_cost: bool = False) \
            -> Tuple[List[ActionAckermannCarSecondOrder], List[ActionAckermannCarSecondOrder]]:
        raise NotImplementedError

    def _propagate_rl_sliding_window(self, q_select: StateAckermannCarSecondOrder, q_rand: StateAckermannCarSecondOrder, radius: float = 0.5,
                                     deterministic_policy: bool = True, add_cost: bool = False, max_steps: int = 20) \
            -> Tuple[List[ActionAckermannCarSecondOrder], List[StateAckermannCarSecondOrder]]:

        _q_rand = copy.copy(q_rand)

        last_state = q_select

        states = []
        actions = []

        for i in range(max_steps):
            adjusted_target_x, adjusted_target_y, adjusted_target_theta, adjusted_target_lin_vel, adjusted_target_phi = self.get_intermediate_target(last_state, q_rand,
                                                                                                  radius)
            (x_start, y_start, theta_start) = transform([last_state.x, last_state.y, last_state.theta], [
                adjusted_target_x, adjusted_target_y, adjusted_target_theta])

            obs = self.env.set_state(x_start, y_start, theta_start, last_state.lin_vel, last_state.phi,
                                     adjusted_target_lin_vel, adjusted_target_phi)

            action, _states = self.model_onnx.predict(
                obs, deterministic=deterministic_policy)

            obs, rewards, done, info = self.env.step(action)
            state = self.env.get_state()
            x_state = state[0]
            y_state = state[1]
            theta_state = state[2]
            lin_vel_state = state[3]
            phi_state = state[4]

            (x, y, theta) = retransform([x_state, y_state, theta_state], [
                adjusted_target_x, adjusted_target_y, adjusted_target_theta])

            q = StateAckermannCarSecondOrder(x, y, theta, lin_vel_state, phi_state)
            states.append(q)
            action = self.dynamics.convert_normalized_action(action)
            u = ActionAckermannCarSecondOrder(action[0], action[1])
            actions.append(u)

            last_state = q

            if done:
                break

        if add_cost:
            assert type(q_select) == CostStateAckermannCarSecondOrder
            q_select: CostStateAckermannCarSecondOrder
            c_init = q_select.cost_to_come
            cstates = [CostStateAckermannCarSecondOrder(s.x, s.y, s.theta, s.lin_vel, s.phi, c_init + (i + 1))
                       for (i, s) in enumerate(states)]
            states = cstates

        return actions, states

    def propagate_mcp(self, s_init: StateAckermannCarSecondOrder, t_sec_min: float, t_sec_max: float, add_cost: bool = False):
        """
        Monte-Carlo Propagation from s_init
        @param s_init: initial state
        @param t_sec_min: minimum time the controls are applied (in sec)
        @param t_sec_max: maximum time the controls are applied (in sec)
        @return: actions, states, duration
        """
        t = np.random.uniform(t_sec_min, t_sec_max)
        action = self.dynamics.sample_action()
        actions, states = self.dynamics.propagate_constant_action(s_init, action, t)
        if add_cost:
            assert type(s_init) == CostStateAckermannCarSecondOrder
            s_init: CostStateAckermannCarSecondOrder
            cost_init = s_init.cost_to_come
            cstates = [CostStateAckermannCarSecondOrder(s.x, s.y, s.theta, s.lin_vel, s.phi, cost_init + (i + 1))
                       for (i, s) in enumerate(states)]
            states = cstates

        return actions, states

    def propagate_mcp_guided(self, s_init: StateAckermannCarSecondOrder, s_target: StateAckermannCarSecondOrder, k: int, t_sec_min: float, t_sec_max: float,
                             add_cost: bool = True):
        """
        Guided monte carlo propagation from s_init towards s_target
        @param t_sec_max:
        @param t_sec_min:
        @param s_init: initial state
        @param s_target: target state
        @param k: number of sampled trajectories
        @return: trajectory which finishes closest to s_target (actions, states, duration)
        """
        min_dist = np.inf
        (best_actions, best_states) = [], []

        for i in range(k):
            _actions, _states = self.propagate_mcp(s_init, t_sec_min, t_sec_max)
            final_state = _states[-1]
            _dist = final_state.distance_to(s_target)
            if _dist < min_dist:
                (best_actions, best_states) = _actions, _states
                min_dist = _dist

        if add_cost:
            assert type(s_init) == CostStateAckermannCarSecondOrder
            s_init: CostStateAckermannCarSecondOrder
            cost_init = s_init.cost_to_come
            best_states = [CostStateAckermannCarSecondOrder(s.x, s.y, s.theta, s.lin_vel, s.phi, cost_init + (i + 1)) for (i, s) in
                           enumerate(best_states)]

        return best_actions, best_states

    @classmethod
    def get_intermediate_target(cls, s_init: StateAckermannCarSecondOrder, s_target: StateAckermannCarSecondOrder, step_size: float, scale_theta: bool = True,
                                scale_lin_vel: bool = True, scale_phi: bool = True) \
            -> Tuple[float, float, float, float, float]:
        """
        Returns an intermediate target based on linear interpolation and a max step size
        @param scale_theta: (bool) flag. if activated (default) theta is scaled according to the ratio between the
                                         translational distance and the step_size
        @param scale_lin_vel: same as scale_theta, but for lin_vel
        @param scale_phi: same as scale_theta, but for phi
        @param s_init: initial state
        @param s_target:  target state
        @param step_size: maximum length of the connection step
        @return: intermediate target (x,y,theta)
        """

        diff_pos = s_init.position_distance_to(s_target)

        if diff_pos < step_size:  # target is close enough
            return s_target.x, s_target.y, s_target.theta, s_target.lin_vel, s_target.phi

        dir_vector = np.asarray((s_target.x - s_init.x, s_target.y - s_init.y)) / diff_pos
        scaled_dir_vector = dir_vector * step_size

        scale = min(step_size / diff_pos, 1)

        if scale_theta:
            theta_diff = s_target.theta - s_init.theta
            direction = 1 if theta_diff > 0 else -1

            if (math.pi * 2 - abs(theta_diff)) < abs(theta_diff):
                theta_diff = (math.pi * 2 - abs(theta_diff))
                direction *= -1

            theta_diff *= scale
            theta_target = (s_init.theta + direction * theta_diff) % (2 * math.pi)
        else:
            theta_target = s_target.theta

        if scale_lin_vel:
            lin_vel_diff = s_target.lin_vel - s_init.lin_vel
            target_lin_vel = s_init.lin_vel + lin_vel_diff * scale
        else:
            target_lin_vel = s_target.lin_vel

        if scale_phi:
            phi_diff = s_target.phi - s_init.phi
            target_phi = s_init.phi + phi_diff * scale
        else:
            target_phi = s_target.phi

        return scaled_dir_vector[0]+s_init.x, scaled_dir_vector[1]+s_init.y, theta_target, target_lin_vel, target_phi
