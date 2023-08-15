import os.path

import yaml
from motion_planning.nearest_neighbour import nearest_neighbor_kdtree as nn
import numpy as np
from motion_planning.collision_checker.collision_checker import CarCollisionChecker2D
from motion_planning.Trees.Node import Node
from spaces.State import StateAckermannCarFirstOrder
from spaces.State import StateAckermannCarSecondOrder
from spaces.State import CostStateAckermannCarFirstOrder
from spaces.State import CostStateAckermannCarSecondOrder
from spaces.State import State
from spaces.Action import Action
from motion_planning.maps.map_2d import Map2D
from dynamics.Dynamics import Dynamics
from driver.driver import Driver
from driver.driver import ExtensionModes
from motion_planning.kdrrt.visualize import Animation
from typing import Callable, Tuple, List, Optional, Union
from motion_planning.motion_planner.MetaMotionPlanner import MetaMotionPlanner
import logging


def _construct_solution(n: Node) -> Tuple[List[Action], List[State]]:
    nodes = []
    temp_n = n
    while temp_n is not None:
        nodes.append(temp_n)
        temp_n = temp_n.parent
    states = []
    actions = []
    for node in reversed(nodes):
        states += node.states
        actions += node.actions
    return actions, states


def generate_intermediate_nodes(parent: Node, actions: List[Action],
                                states: List[State], states_per_node=10) -> List[Node]:
    nodes = []
    first_node = Node(parent, actions[:states_per_node], states[:states_per_node])
    nodes.append(first_node)
    last_max_index = states_per_node

    i = 2
    while last_max_index < len(states):
        _max_index = i * states_per_node
        _node = Node(nodes[-1], actions[last_max_index:_max_index], states[last_max_index:_max_index])
        nodes.append(_node)
        i += 1
        last_max_index = _max_index
    return nodes


class RRTPlanner(MetaMotionPlanner):

    def __init__(self, config_path: str, dynamics: Dynamics, map2d: Map2D, collision_checker: CarCollisionChecker2D,
                 goal_state_evaluator: Callable,
                 s_init: Optional[Union[StateAckermannCarFirstOrder, StateAckermannCarSecondOrder]] = None):
        super().__init__(
            planner_id="RRT",
            config_path=config_path,
            planner_dir=os.path.dirname(__file__)
        )

        # load configs (No defaults)
        self.config = yaml.safe_load(open(config_path, "r"))
        self.PROPAGATION_T_MIN = self.config["propagation_t_min"]
        self.PROPAGATION_T_MAX = self.config["propagation_t_max"]
        self.NN_STRUCT_MAX_WAITING_NODES = self.config["nn_struct_waiting_nodes"]
        self.MAX_TRIES_COLLISION_FREE_SAMPLING = self.config["max_tries_collision_free_sampling"]
        self.GOAL_BIAS = self.config["goal_bias"]

        self.nn_struct = nn.NNStructure(self.NN_STRUCT_MAX_WAITING_NODES)

        self.map = map2d
        self.dynamics = dynamics

        self.collision_checker = collision_checker

        self.s_init = s_init

        self.n_root = Node(
            parent=None,
            actions=[],
            states=[self.s_init]
        )

        self.nn_struct.add_node(self.n_root)
        self.goal_state_evaluator = goal_state_evaluator
        self.logger = logging.getLogger(__file__)

    def visualize_trajectory(self,
                             states: Union[List[StateAckermannCarFirstOrder], List[StateAckermannCarSecondOrder]]):
        visualizer = Animation(self.map.map_path, q_start_overwrite=None, q_goal_overwrite=None)
        visualizer.add_trajectory(states)
        visualizer.show()

    def _reset(self):
        self.nn_struct = nn.NNStructure(self.NN_STRUCT_MAX_WAITING_NODES)
        self.nn_struct.add_node(self.n_root)

    def state_has_feasible_cost(self, s: CostStateAckermannCarFirstOrder, cost_boundary: float):
        if s.cost_to_come > cost_boundary:
            return False
        return True

    def _is_free_traj(self, states: List[State], cost_boundary=np.inf):
        """
        Checks if a given List of car states collides with the obstacles in the current map
        @param states: List of states
        @return: True if free, else False
        """
        assert issubclass(type(states[0]), StateAckermannCarFirstOrder)
        states: List[StateAckermannCarFirstOrder]
        for p in states:
            if self.collision_checker.collides(p.x, p.y, p.theta):
                return False

            if type(p) == CostStateAckermannCarFirstOrder:  # if the state has a cost attached, check if it violates
                p: CostStateAckermannCarFirstOrder
                if not self.state_has_feasible_cost(p, cost_boundary):
                    return False

            if type(p) == CostStateAckermannCarSecondOrder:  # if the state has a cost attached, check if it violates
                p: CostStateAckermannCarSecondOrder
                if not self.state_has_feasible_cost(p, cost_boundary):
                    return False

        return True

    def sample_valid_state(self) -> State:
        k = 0
        q_rand = self.dynamics.sample_valid_state(scenario_map=self.map)
        while self.collision_checker.collides(*q_rand.vector()[:3]):
            if k > self.MAX_TRIES_COLLISION_FREE_SAMPLING:
                raise Exception("Struggling to find collision free samples on the map...")
            q_rand = self.dynamics.sample_valid_state(scenario_map=self.map)
            k += 1
        return q_rand

    def find_trajectory_to(self, destination: State, extension_mode: ExtensionModes,
                           driver: Driver, max_iterations: int,
                           intermediate_nodes: bool,
                           states_per_node: int,
                           cost_boundary: float = np.inf, ao_rrt=False):
        """
        @param states_per_node:
        @param destination: Target State
        @param extension_mode: Method used to extend from the selected node (mcp, guided mcp, rl, rl-onnx)
        @param driver: Driver used for the extension
        @param max_iterations: maximum number of samples
        @param cost_boundary: [use this only for AO-RRT] defines a cost boundary. nodes above this will not be added to the tree
        @param intermediate_nodes: (bool) if True, nodes along the trajectory of a rollout will be added to the tree
        @return: (actions, states) if no path could be found: [], []
        """
        # assert type(destination) == StateAckermanFirstOrder
        assert issubclass(type(destination), StateAckermannCarFirstOrder)
        assert self.s_init is not None

        if ao_rrt:
            if type(destination) == StateAckermannCarFirstOrder:
                destination = CostStateAckermannCarFirstOrder(destination.x, destination.y, destination.theta,
                                                              cost_to_come=cost_boundary)
            elif type(destination) == StateAckermannCarSecondOrder:
                destination = CostStateAckermannCarSecondOrder(destination.x, destination.y, destination.theta,
                                                               destination.lin_vel, destination.phi,
                                                               cost_to_come=cost_boundary)

        rounds_left = max_iterations
        c = 0
        while rounds_left != 0:
            rounds_left -= 1
            if c % 100 == 0:
                self.logger.info(f"c : {c} nodes: {len(self.nn_struct.nodes)}")

            if c % 1000 == 0:
                q_closest_to_goal = self.nn_struct.get_nearest_node(destination)
                _min_dist = q_closest_to_goal.get_final_state().distance_to(destination)
                self.logger.info(f"closest node: state: {q_closest_to_goal.get_final_state().vector()}")
                self.logger.info(f"dist: {_min_dist} [extends: {q_closest_to_goal.extends}")
                self.logger.info(f"(failed extends: {q_closest_to_goal.failed_extends}")
                self.logger.info(f"| ratio: {q_closest_to_goal.get_extension_failure_ratio()})]")

            x = np.random.uniform(0, 1)

            if x < self.GOAL_BIAS:  # use the destination as the expansion target
                if cost_boundary == np.inf:
                    # in the first round, c_max is set to the maximum cost in the tree
                    destination.cost_to_come = self.nn_struct.max_cost
                q_rand = destination
            else:  # sample random (non-colliding) expansion target
                q_rand = self.sample_valid_state()
                if ao_rrt:
                    if cost_boundary == np.inf:
                        sample_cost = np.random.uniform(0, self.nn_struct.max_cost)
                    else:
                        sample_cost = np.random.uniform(0, cost_boundary)
                    if type(q_rand) == StateAckermannCarFirstOrder:
                        q_rand: StateAckermannCarFirstOrder
                        q_rand = CostStateAckermannCarFirstOrder(q_rand.x, q_rand.y, q_rand.theta, sample_cost)
                    elif type(q_rand) == StateAckermannCarSecondOrder:
                        q_rand: CostStateAckermannCarSecondOrder
                        q_rand = CostStateAckermannCarSecondOrder(q_rand.x, q_rand.y, q_rand.theta, q_rand.lin_vel, q_rand.phi, sample_cost)

            n_select = self.nn_struct.get_nearest_node(q_rand)

            if extension_mode == ExtensionModes.RL:
                # RL-agent
                actions, states = driver.propagate_rl(n_select.get_final_state(), q_rand, add_cost=True)

            elif extension_mode == ExtensionModes.GUIDED_MCP:
                # monte-carlo propagation, but select the closest from a bunch of sampled trajectories
                nr_propagation_samples = self.config["mcp_propagation_samples_one"]
                actions, states = driver.propagate_mcp_guided(
                    n_select.get_final_state(), q_rand, nr_propagation_samples,
                    self.PROPAGATION_T_MIN, self.PROPAGATION_T_MAX, add_cost=ao_rrt)
            elif extension_mode == ExtensionModes.RL_ONNX:
                actions, states = driver.propagate_rl(n_select.get_final_state(), q_rand, use_onnx_model=True,
                                                      add_cost=True, sliding_window_mode=True)
            else:
                # default: unguided mcp
                actions, states = driver.propagate_mcp(n_select.get_final_state(),
                                                       self.PROPAGATION_T_MIN, self.PROPAGATION_T_MAX,
                                                       add_cost=True)

            n_select.extends += 1

            if self._is_free_traj(states, cost_boundary):
                new_node = Node(n_select, actions, states)
                if intermediate_nodes:
                    new_nodes = generate_intermediate_nodes(n_select, actions, states, states_per_node)
                    for node in new_nodes:
                        self.nn_struct.add_node(node)
                else:
                    self.nn_struct.add_node(new_node)
                if self.goal_state_evaluator(new_node.get_final_state(), destination):
                    actions, states = _construct_solution(new_node)
                    return actions, states
            else:
                n_select.failed_extends += 1

            c += 1

        return [], []

    def simple_prune(self, cost_boundary):
        new_nodes = [n for n in self.nn_struct.nodes if n.get_final_state().cost_to_come < cost_boundary]
        self.nn_struct.nodes = new_nodes
        self.nn_struct.build_tree()
