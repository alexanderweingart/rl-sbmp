import os.path
import datetime
import random
import logging

import git
import numpy as np
from rich.logging import RichHandler
from dynamics.Dynamics import Dynamics
import yaml
from motion_planning.collision_checker.collision_checker import CarCollisionChecker2D
from driver.driver import Driver
from motion_planning.motion_planner import MetaMotionPlanner
from motion_planning.kdrrt.kdrrt import RRTPlanner
from motion_planning.kdrrt.aorrt import AORRTPlanner
from motion_planning.maps.map_2d import Map2D
from spaces.State import State
from typing import Callable, Optional
from spaces.Action import Action
from driver.driver import ExtensionModes
from dynamics.AckermannCarDynamicsFirstOrder import AckermannCarDynamicsFirstOrder
from driver.ackermann_driver import AckermannCarFirstOrderDriver
from driver.ackermann_driver import AckermannCarSecondOrderDriver
from motion_planning.collision_checker.collision_checker import CarCollisionChecker2D
from spaces.State import StateAckermannCarFirstOrder
from spaces.State import StateAckermannCarSecondOrder
from spaces.State import CostStateAckermannCarFirstOrder
from spaces.State import CostStateAckermannCarSecondOrder
from argparse import ArgumentParser
from rich import print
from motion_planning.utils.TrajectoryHandler import TrajectoryHandler
from dynamics.AckermannCarDynamicsSecondOrder import AckermannCarDynamicsSecondOrder
import time
from utils.repo_checker import RepoChecker





def seed_all_rnd_generators(new_seed: int):
    np.random.seed(new_seed)
    random.seed(new_seed)


class MotionPlanningEvaluator:
    available_planners = ["rrt", "ao-rrt"]
    available_dynamics = ["ackermann_car_first_order"]

    def __init__(self, planner_id, dynamics: Dynamics, driver: Driver, collision_checker: CarCollisionChecker2D,
                 map_path: str, planner_config_path: str, data_dir,
                 goal_state_evaluator, intermediate_nodes, states_per_node, max_iterations,
                 extension_mode: ExtensionModes, s_init: State, intermediate_traj_dir: str):

        self.intermediate_nodes = intermediate_nodes
        self.states_per_node = states_per_node
        self.max_iterations = max_iterations
        self.extension_mode = extension_mode
        self.data_dir = data_dir
        self.driver = driver
        self.dynamics = dynamics

        map2d = Map2D(map_path)
        assert planner_id in self.available_planners

        if planner_id == "rrt":
            self.planner = RRTPlanner(
                config_path=planner_config_path,
                dynamics=dynamics,
                collision_checker=collision_checker,
                goal_state_evaluator=goal_state_evaluator,
                map2d=map2d,
                s_init=s_init
            )
        elif planner_id == "ao-rrt":
            self.planner = AORRTPlanner(
                config_path=planner_config_path,
                dynamics=dynamics,
                collision_checker=collision_checker,
                goal_state_evaluator=goal_state_evaluator,
                map2d=map2d,
                s_init=s_init,
                intermediate_traj_dir=intermediate_traj_dir
            )

        else:
            raise NotImplementedError

    def solve(self, s_target: State, timeout_secs: float, intermediate_traj_dir: Optional[str] = None):
        return self.planner.find_trajectory_to(
            destination=s_target,
            extension_mode=self.extension_mode,
            driver=self.driver,
            max_iterations=self.max_iterations,
            intermediate_nodes=self.intermediate_nodes,
            states_per_node=self.states_per_node,
            timeout_secs=timeout_secs
        )

    @classmethod
    def from_config(cls, planner_config_path: str, robot_config_path: str, problem_config_path: str,
                    data_dir: str, planner_id: str, timestamp_str: str, generate_intermediate_trajs: bool = False):
        planner_config = yaml.safe_load(open(planner_config_path, "r"))
        intermediate_nodes = planner_config["intermediate_nodes"]
        if intermediate_nodes:
            states_per_node = planner_config["states_per_node"]
        else:
            states_per_node = None

        max_iterations = planner_config["max_iterations"]
        extension_mode = planner_config["extension_mode"]

        problem_config = yaml.safe_load(open(problem_config_path, "r"))

        goal_tolerance_orientation = problem_config["goal_tolerance_theta"]
        goal_tolerance_position = problem_config["goal_tolerance_pos"]

        start_state_vector = problem_config["start"]

        robot_config = yaml.safe_load(open(robot_config_path, "r"))
        car_length = robot_config["L"]
        car_width = robot_config["W"]

        map_path = problem_config["map_path"]
        dynamics = problem_config["dynamics"]

        if dynamics not in planner_config["models"]:
            raise ValueError

        if "onnx" in planner_config["models"][dynamics]:
            model_path_onnx = planner_config["models"][dynamics]["onnx"]
        else:
            model_path_onnx = None

        if "sb3" in planner_config["models"][dynamics]:
            model_path = planner_config["models"][dynamics]["sb3"]
        else:
            model_path = None

        collision_checker = CarCollisionChecker2D(car_length=car_length, car_width=car_width)
        collision_checker.load_obstacles(map_path)

        if dynamics == "ackermann_car_first_order":
            dynamics = AckermannCarDynamicsFirstOrder(robot_config_path)
            driver = AckermannCarFirstOrderDriver(config_path=robot_config_path,
                                                  model_path_onnx=model_path_onnx, model_path=model_path,
                                                  dynamics=dynamics)

            s_init = CostStateAckermannCarFirstOrder(*start_state_vector, 0)

            def goal_state_evaluator(s: StateAckermannCarFirstOrder, s_goal: StateAckermannCarFirstOrder):
                d = s.position_distance_to(s_goal)
                if d > goal_tolerance_position:
                    return False
                d_o = s.orientation_distance_to(s_goal)
                if d_o > goal_tolerance_orientation:
                    return False
                return True

        elif dynamics == "ackermann_car_second_order":
            dynamics = AckermannCarDynamicsSecondOrder(robot_config_path)
            driver = AckermannCarSecondOrderDriver(config_path=robot_config_path, model_path_onnx=model_path_onnx,
                                                   model_path=model_path, dynamics=dynamics)
            s_init = CostStateAckermannCarSecondOrder(*start_state_vector, 0)
            goal_tolerance_lin_vel = problem_config["goal_tolerance_lin_vel"]
            goal_tolerance_phi = problem_config["goal_tolerance_phi"]

            def goal_state_evaluator(s: StateAckermannCarSecondOrder, s_goal: StateAckermannCarSecondOrder):
                d = s.position_distance_to(s_goal)
                if d > goal_tolerance_position:
                    return False
                d_o = s.orientation_distance_to(s_goal)
                if d_o > goal_tolerance_orientation:
                    return False
                diff_lin_vel = abs(s_goal.lin_vel - s.lin_vel)
                if diff_lin_vel > goal_tolerance_lin_vel:
                    return False
                diff_phi = abs(s_goal.phi - s_goal.phi)
                if diff_phi > goal_tolerance_phi:
                    return False
                return True
        else:
            raise ValueError()

        return cls(
            planner_id=planner_id,
            dynamics=dynamics,
            driver=driver,
            collision_checker=collision_checker,
            map_path=map_path,
            planner_config_path=planner_config_path,
            data_dir=data_dir,
            intermediate_nodes=intermediate_nodes,
            states_per_node=states_per_node,
            max_iterations=max_iterations,
            extension_mode=ExtensionModes[extension_mode],
            goal_state_evaluator=goal_state_evaluator,
            s_init=s_init,
            intermediate_traj_dir=data_dir
        )

    def store_nodes(self, output_path: str):
        node_dicts = [{
            "id": id(n),
            "parent": id(n.parent),
            "states": [s.vector() for s in n.states],
            "actions": [s.vector() for s in n.actions]
        } for n in self.planner.nn_struct.nodes]
        ret_dir = {
            "nodes": node_dicts
        }
        yaml.safe_dump(ret_dir, open(output_path, "w"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("planner", type=str, choices=MotionPlanningEvaluator.available_planners)
    parser.add_argument("--config", "-c", type=str)
    parser.add_argument("--output_dir", "-o", type=str,
                        default=os.path.join(os.path.dirname(__file__), "experiment_data"))
    parser.add_argument("--skip_git_check", "-sgc", action="store_true")
    parser.add_argument("--daemon", "-d", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # logging setup:


    handlers = []
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{timestamp}.log")
    if args.verbose:
        logging_level = logging.INFO
    else:
        logging_level = logging.WARNING

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(fh_formatter)
    handlers.append(file_handler)

    if not args.daemon:
        handlers.append(RichHandler(level=logging_level, show_time=True, omit_repeated_times=False))

    logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%x %X]", handlers=handlers)
    logger = logging.getLogger(__file__)

    ############
    logger.info("Starting the evaluation")
    logger.info("args:")
    logger.info(args.__dict__)

    repo_checker = RepoChecker()
    commit_dir_name = repo_checker.get_commit_dir_name()
    if repo_checker.is_dirty():
        if not args.skip_git_check:
            logger.warning("Aborting Evaluation, because the repo is dirty. If you want to keep going anyways, set"
                           "the --skip_git_check flag")
            raise Exception("The repository contains uncommited changes. You risk creating non-reproductible data.")
        commit_dir_name = "dirty_repo"

    config = yaml.safe_load(open(args.config, "r"))

    if 'nr_of_runs' in config:
        nr_of_runs = config['nr_of_runs']
    else:
        nr_of_runs = 1

    if 'seeds' in config:
        seeds = config['seeds']
    else:
        seeds = []

    if 'timeout_secs' in config:
        timeout_secs = config['timeout_secs']
    else:
        timeout_secs = None

    planner_config_path = config["planner_config_path"]
    problem_config_path = config["problem_config_path"]
    robot_config_path = config["robot_config_path"]
    store_intermediate_trajs = config["store_intermediate_trajs"]

    experiment_config_name = args.config.split("/")[-1].split(".")[0]

    for i in range(nr_of_runs):
        run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger.info(f"###### ROUND {i} ######")

        if i < len(seeds):
            seed = seeds[i]
        else:
            seed = int(time.time())

        seed_all_rnd_generators(seed)

        os.makedirs(args.output_dir, exist_ok=True)
        experiment_dir = os.path.join(args.output_dir, commit_dir_name, experiment_config_name, run_timestamp)
        os.makedirs(experiment_dir, exist_ok=True)

        meta_data = {
            "seed": seed,
        }

        yaml.safe_dump(meta_data, open(os.path.join(experiment_dir, "meta.yaml"), "w"))

        mp = MotionPlanningEvaluator.from_config(
            planner_id=args.planner,
            planner_config_path=planner_config_path,
            robot_config_path=robot_config_path,
            problem_config_path=problem_config_path,
            data_dir=experiment_dir,
            timestamp_str=timestamp,
            generate_intermediate_trajs=store_intermediate_trajs
        )

        problem_config = yaml.safe_load(open(problem_config_path, "r"))

        if problem_config["dynamics"] == "ackermann_car_first_order":
            s_target = StateAckermannCarFirstOrder(*problem_config["goal"])
        elif problem_config["dynamics"] == "ackermann_car_second_order":
            s_target = StateAckermannCarSecondOrder(*problem_config["goal"])
        else:
            logger.warning(f"the specified dynamics ({problem_config['dynamics']}) are not supported yet. --> aborting")
            raise NotImplementedError

        logger.info("Starting the solver.")
        t_beginning = datetime.datetime.now()
        actions, states = mp.solve(s_target, timeout_secs, experiment_dir)
        t_end = datetime.datetime.now()
        logger.warning(f"solve runtime: {(t_end-t_beginning).total_seconds():.4f}")

        if len(actions) == 0 or len(states) == 0:
            logging.warning("Solver failed to find feasible motion plan.")
        else:
            traj_path = os.path.join(experiment_dir, f"final_trajectory_{run_timestamp}.yaml")
            traj_dict = TrajectoryHandler.store_trajectory(mp.planner.dynamics, actions, states, s_target, traj_path)
            logger.info(f"new trajectory store to {traj_path}")

            print("_______________________________")
            print("[green] Found a trajectory!")
            duration = t_end - t_beginning
            print(f"[start: {timestamp}] -> [end: {t_end.strftime('%Y-%m-%d_%I-%M-%S')}] [{duration.total_seconds():.3f}]")
            print("_______________________________")
            print(f"> duration:         {traj_dict['t_total_sec']}")
            print(f"> final distance:   {traj_dict['final_dist']}")
            print("_______________________________")

