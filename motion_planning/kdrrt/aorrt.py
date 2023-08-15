import datetime
import logging

from motion_planning.kdrrt.kdrrt import RRTPlanner
import os
from dynamics.Dynamics import Dynamics
from motion_planning.maps.map_2d import Map2D
from motion_planning.collision_checker.collision_checker import CarCollisionChecker2D
from spaces.State import StateAckermannCarFirstOrder
from spaces.State import StateAckermannCarSecondOrder
from spaces.State import CostStateAckermannCarFirstOrder
from spaces.State import CostStateAckermannCarSecondOrder
from spaces.State import State
from driver.ackermann_driver import AckermannCarFirstOrderDriver
from driver.driver import ExtensionModes
from typing import Callable, Optional
import numpy as np
from driver.driver import Driver
from motion_planning.utils.TrajectoryHandler import TrajectoryHandler
from multiprocessing import Process, Manager


class AORRTPlanner(RRTPlanner):
    def __init__(self, config_path: str, dynamics: Dynamics, map2d: Map2D, collision_checker: CarCollisionChecker2D,
                 goal_state_evaluator: Callable, s_init: Optional[StateAckermannCarFirstOrder] = None,
                 intermediate_traj_dir: Optional[str] = None, visualize_each_iteration: bool = False):
        super().__init__(config_path, dynamics, map2d, collision_checker, goal_state_evaluator, s_init)
        self.intermediate_traj_dir = intermediate_traj_dir
        os.makedirs(self.intermediate_traj_dir, exist_ok=True)
        self.visualize_each_iteration = visualize_each_iteration
        self.reuse_tree = self.config["reuse_tree"]
        self.logger = logging.getLogger(__file__)
        self.cost_boundary_shrink_factor = self.config["cost_boundary_shrink_factor"]

    def aorrt_main_loop(self, rounds, cost_boundary, destination, extension_mode, driver, max_iterations, intermediate_nodes, states_per_node, max_failures, best_actions, best_states):
        failure_count = 0
        i = rounds
        round = 0
        while i != 0:  # this way, setting rounds to -1 lead to (quasi-)open ended iterations
            i -= 1
            round += 1
            self.logger.info(f"Round {'(open ended)' if i < 0 else ''}{abs(i) if rounds < 0 else rounds - i}  "
                             f"--- Cost Boundary: {cost_boundary}")

            pre_nodes = len(self.nn_struct.nodes)
            self.simple_prune(cost_boundary)
            post_nodes = len(self.nn_struct.nodes)
            self.logger.info(f"pruning: [{pre_nodes}] -> [{post_nodes}])")
            t_start = datetime.datetime.now()
            actions, states = super().find_trajectory_to(
                destination=destination,
                extension_mode=extension_mode,
                driver=driver,
                max_iterations=max_iterations,
                intermediate_nodes=intermediate_nodes,
                states_per_node=states_per_node,
                cost_boundary=cost_boundary,
                ao_rrt=True
            )
            t_finished = datetime.datetime.now()

            if actions == [] or states == []:
                failure_count += 1
                if failure_count > max_failures and max_failures >= 0:  # if max_failures is set to a nr < 0, ignore the failures
                    self.logger.info("find trajectory was not successful anymore. returning the current best.")
                    return best_actions, best_states

            else:
                self.logger.info("find trajectory was successful!")
                failure_count = 0
                best_states[:] =  []
                best_actions[:] = []
                for action in actions:
                    best_actions.append(action)
                for state in states:
                    best_states.append(state)
                best_states: list[CostStateAckermannCarFirstOrder]

                if self.visualize_each_iteration:
                    self.visualize_trajectory(best_states)

                if self.intermediate_traj_dir is not None:
                    TrajectoryHandler.store_trajectory(
                        t_start=t_start,
                        t_finished=t_finished,
                        dynamics=self.dynamics,
                        actions=actions,
                        states=states,
                        s_target=destination,
                        output_path=os.path.join(self.intermediate_traj_dir, f"intermediate_trajectory_{round}.yaml"),
                        map_path=self.map.map_path
                    )

            cost_boundary = best_states[-1].cost_to_come * self.cost_boundary_shrink_factor
            if not self.reuse_tree:
                self.n_root.failed_extends = 0
                self.n_root.extends = 0
                self.nn_struct.nodes = [self.n_root]
                self.nn_struct.build_tree()

        return best_actions, best_states

    def find_trajectory_to(self, destination: State, extension_mode: ExtensionModes,
                           driver: Driver, max_iterations: int,
                           intermediate_nodes: bool,
                           states_per_node: int,
                           timeout_secs: float,
                           cost_boundary: float = np.inf):
        max_failures = self.config["max_failures"]
        rounds = self.config["rounds"]
        max_iterations = self.config["max_iterations"]
        extension_mode = ExtensionModes[self.config["extension_mode"]]

        # initialize the first state with cost 0
        if type(self.s_init) == StateAckermannCarFirstOrder:
            self.s_init = CostStateAckermannCarFirstOrder(self.s_init.x, self.s_init.y, self.s_init.theta, 0)
        elif type(self.s_init) == StateAckermannCarSecondOrder:
            self.s_init = CostStateAckermannCarSecondOrder(self.s_init.x, self.s_init.y, self.s_init.theta,
                                                           self.s_init.lin_vel, self.phi, 0)

        manager = Manager()
        best_actions = manager.list()
        best_states = manager.list()
        main_loop_process = Process(target=self.aorrt_main_loop, args=(rounds,np.inf,destination,extension_mode,driver,max_iterations,intermediate_nodes,states_per_node,max_failures,best_actions,best_states))
        main_loop_process.start()
        main_loop_process.join(timeout=timeout_secs)
        exit_code = main_loop_process.exitcode
        if exit_code != 0:
            logging.info("process did not terminate yet. shutting down.")
            main_loop_process.terminate()


        return best_actions, best_states
