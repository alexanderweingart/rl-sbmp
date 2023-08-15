import argparse
import matplotlib.pyplot as plt
import visualize
import importlib
from rich.console import Console
from rich.logging import RichHandler
import git
import hashlib
import logging
import logging.config
import datetime
import numpy as np
import os
import time
import yaml
import subprocess
# from scenario_config import Config, collision, sample_config_uniform, ControlSpace
from world_utils import Rectangle, Workspace
from motion_planning.kdrrt.kdrrt import RRTPlanner
from spaces.State import State
from spaces.Action import Action
from motion_planning.maps.map_2d import Map2D
from motion_planning.collision_checker.collision_checker import CarCollisionChecker2D
from dynamics.AckermannCarDynamicsFirstOrder import AckermannCarDynamicsFirstOrder
from spaces.State import StateAckermannCarFirstOrder
from spaces.State import State
from driver.ackermann_driver import AckermannCarFirstOrderDriver
from dynamics.AckermannCarDynamicsSecondOrder import AckermannCarDynamicsSecondOrder
from driver.driver import ExtensionModes
from rl.environments.car import CustomEnv
import datetime
from spaces.State import CostStateAckermannCarFirstOrder


def write_header_to_result_file(result_file_path: str):
    if not os.path.exists(result_file_path):
        with open(result_file_path, "w") as f:
            f.write(
                "t,approach,t_diff,nr_of_nodes,success \n")


def write_results_to_result_file(result_file_path: str, t_diff: float, approach: str,
                                 nr_of_nodes: int, success: bool):
    with open(result_file_path, "a") as f:
        f.write(
            f"{datetime.datetime.now()},{approach},{t_diff},{nr_of_nodes},{success}\n")


def write_traj_to_file(experiment_sub_dir_path: str, states: list[State], actions: list[Action],
                       algo: str, dt: float):
    os.makedirs(os.path.join(experiment_sub_dir_path, "trajectories"), exist_ok=True)
    file_name = os.path.join(experiment_sub_dir_path, "trajectories", f"{time.time()}_{algo}.yaml")

    traj_array = [s.vector().tolist() for s in states]
    controls_array = [a.vector().tolist() for a in actions]
    with open(file_name, "w") as f:
        traj_dict = {
            "algo": algo,
            "dt": dt,
            "states": traj_array,
            "actions": controls_array
        }
        yaml.safe_dump(traj_dict, f)


def dump_args(args, output_dir):
    with open(output_dir, "w") as f:
        f.write(yaml.dump(args))


def generate_hash(config_file_path, project_files=None):
    if project_files is None:
        # project_files = ["kdrrt_evaluator.py", "kdrrt.py", "dubin_vel_control.py",
        #                  "dubin_vel_control_lin_vel_fixed.py", "nearest_neighbor.py",
        #                  "scenario_config.py", "project_utils.py", "dubins_car.py"]
        raise NotImplementedError
    file_hash = hashlib.sha256()  # Create the hash object, can use something other than `.sha256()` if you wish
    BLOCK_SIZE = 65536  # The size of each read from the file
    files = project_files + [config_file_path]
    for file in files:
        with open(file, 'rb') as f:  # Open the file to read it's bytes
            fb = f.read(BLOCK_SIZE)  # Read from the file. Take in the amount declared above
            while len(fb) > 0:  # While there is still data being read from the file
                file_hash.update(fb)  # Update the hash
                fb = f.read(BLOCK_SIZE)  # Read the next block from the file

    return file_hash.hexdigest()  # Get the hexadecimal digest of the hash


def gen_experiment_name(experiment_label: str, config_file_path: str, hash_portion_len=5):
    # TODO: switched to datetime name for now. the hash version was cooler, but is currently not necessary

    # project_file_hash = generate_hash(config_file_path)
    # if len(project_file_hash) > hash_portion_len:
    #     project_file_hash = project_file_hash[0:hash_portion_len]
    # return experiment_label + "_" + project_file_hash
    return f"{experiment_label}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"


def check_if_project_files_changed(config_file_path, project_files=None, repo_root="."):
    raise NotImplementedError("The git check is currently broken by the restructuring. TODO for later")
    # if project_files is None:
    #     project_files = ["kdrrt_evaluator.py", "kdrrt.py", "dubin_vel_control.py",
    #                      "dubin_vel_control_lin_vel_fixed.py",
    #                      "nearest_neighbor.py", "scenario_config.py", "project_utils.py",
    #                      "dubins_car.py"]
    # project_files += [config_file_path]
    # repo = git.Repo(repo_root)
    # git_relative_project_file_paths = [os.path.join("code", "kdrrt", file_path) for file_path in project_files]
    # dirty_file_paths = [item.a_path for item in repo.index.diff(None)]
    # dirty_project_files = [path for path in git_relative_project_file_paths if path in dirty_file_paths]
    # not_tracked_project_files = [path for path in git_relative_project_file_paths if path in repo.untracked_files]
    #
    # if len(dirty_project_files) > 0 or len(not_tracked_project_files) > 0:
    #     logging.warning(
    #         f"relevant project files are dirty or untracked \n\t>dirty: {dirty_project_files})\n\t>untracked: {not_tracked_project_files}")
    #     return False
    # return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("system", choices=["lin_vel_ctrl", "accl_ctrl"])
    parser.add_argument("-c", "--candidates", nargs="+")
    parser.add_argument("--experiment_dir", default="evaluations",
                        help="path to the directory you want to store the experiment results in")
    parser.add_argument("--map_path", dest="map_path", default="./example0_map_small.yaml")
    parser.add_argument("--random_locations", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--goal_bias", default=0.1)
    parser.add_argument("--seed", required=False)
    parser.add_argument("--t_min_mcp", help="minimum trajectory time for mcp", required=False)
    parser.add_argument("--t_max_mcp", help="minimum trajectory time for mcp", required=False)
    parser.add_argument("--model_path_onnx", required=False, type=str)
    parser.add_argument(
        "--start_pose", type=tuple[float, float, float], required=False)
    parser.add_argument(
        "--end_pose", type=tuple[float, float, float], required=False)
    parser.add_argument("--robot_config", type=str, required=False)
    parser.add_argument("--n_mcp", required=False, default=50)
    parser.add_argument("--only_mcp", action="store_true")
    parser.add_argument("--goal_tolerance_pos", default=0.5)
    parser.add_argument("--goal_tolerance_theta", default=0.2 * np.pi)
    parser.add_argument("--experiment_label", default="nolabel")
    parser.add_argument("--skip_git_check", action="store_true")
    parser.add_argument("--scenario_config", default="scenario_config.py")
    parser.add_argument("--rl_driver", required=False)
    parser.add_argument("--store_traj_plot", action="store_true")
    parser.add_argument("--model_path", required=False)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--optimal", action="store_true")
    args = parser.parse_args()

    logging.config.fileConfig("motion_planning/kdrrt/log.ini")
    logger = logging.getLogger(__file__)
    logger.addHandler(RichHandler())

    if args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    os.makedirs("logs", exist_ok=True)

    console = Console()

    if args.config is not None:  # config params overwrite command line args
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)
            for key in config.keys():
                args.__dict__[key] = config[key]

    scenario_conf = yaml.safe_load(open(args.map_path, "r"))

    (x_init, y_init, theta_init) = scenario_conf["robot"]["start"]
    (x_goal, y_goal, theta_goal) = scenario_conf["robot"]["goal"]

    time_stamp = time.time()

    if args.experiment_dir != "." and not os.path.exists(args.experiment_dir):
        input(f"Creating {args.experiment_dir} now. Press Enter to continue.")
        os.makedirs(args.experiment_dir)

    experiment_sub_dir = os.path.join(
        os.getcwd(), args.experiment_dir, gen_experiment_name(args.experiment_label, args.config))

    os.makedirs(experiment_sub_dir, exist_ok=True)
    logging.info(f"experiment_sub_dir: {experiment_sub_dir}")

    root_scenario_path_local = os.path.join(
        experiment_sub_dir, "root_scenario.yaml")

    #  safe relevant project files
    subprocess.run(args=["cp", args.map_path, root_scenario_path_local])

    robot_config = yaml.safe_load(open(args.robot_config, "r"))

    car_width = robot_config["W"]
    car_length = robot_config["L"]
    map2d = Map2D(args.map_path)
    collision_checker = CarCollisionChecker2D(
        car_length=car_length,
        car_width=car_width,
        inflation=1
    )
    collision_checker.load_obstacles(root_scenario_path_local)

    if args.system == "lin_vel_ctrl":
        from dynamics.AckermannCarDynamicsFirstOrder import AckermannCarDynamicsFirstOrder as Dynamics
        from spaces.State import StateAckermannCarFirstOrder as State
        from spaces.Action import ActionAckermannCarFirstOrder as Action
    else:
        from dynamics.AckermannCarDynamicsFirstOrder import AckermannCarDynamicsFirstOrder as Dynamics
        from spaces.State import StateAckermannCarSecondOrder as State
        from spaces.Action import ActionAckermannCarSecondOrder as Action

    dynamics = Dynamics(args.robot_config)

    kdrrt_planner = RRTPlanner(
        dynamics=dynamics,
        map2d=map2d,
        collision_checker=collision_checker,
        s_init=None,
        propagation_t_min=args.t_min_mcp,
        propagation_t_max=args.t_max_mcp,
        goal_tolerance_pos=args.goal_tolerance_pos,
        goal_tolerance_theta=args.goal_tolerance_theta
    )

    with open(root_scenario_path_local, "r") as f:
        root_scenario_conf = yaml.safe_load(f)

    if args.random_locations:
        scenarios = []

        for i in range(args.n):
            s_init = kdrrt_planner.sample_valid_state()
            s_goal = kdrrt_planner.sample_valid_state()

            assert type(s_init) == StateAckermannCarFirstOrder | StateAckermannCarFirstOrder
            s_init: StateAckermannCarFirstOrder
            s_goal: StateAckermannCarFirstOrder

            scneario_conf = root_scenario_conf.copy()
            scenario_conf["robot"]["start"] = [
                s_init.x, s_init.y, s_init.theta]
            scenario_conf["robot"]["goal"] = [s_goal.x, s_goal.y, s_goal.theta]
            scenario_conf_path = os.path.join(
                experiment_sub_dir, f"{time_stamp}_rand_conf_{i}.yaml")
            with open(scenario_conf_path, "w") as f:
                f.write(yaml.safe_dump(scenario_conf))
            scenarios.append(scenario_conf)
    else:
        if args.start_pose is not None:
            root_scenario_conf["robot"]["start"] = args.start_pose
        if args.end_pose is not None:
            root_scenario_conf["robot"]["goal"] = args.end_pose

        scenario_conf_path = os.path.join(
            experiment_sub_dir, f"custom_conf_{time_stamp}.yaml")
        with open(scenario_conf_path, "w") as f:
            f.write(yaml.safe_dump(scenario_conf))

        scenarios = [root_scenario_conf]

    # only relevant to the random control case
    results = []

    result_path = os.path.join(experiment_sub_dir, "results.csv")
    write_header_to_result_file(result_path)

    args_path = os.path.join(experiment_sub_dir, "args.yaml")
    dump_args(args, args_path)

    for i in range(args.n):
        logging.info(f"---- Round {i} ----")
        if args.random_locations:
            scenario = scenarios[i]
        else:
            scenario = root_scenario_conf

        (x_init, y_init, theta_init) = scenario_conf["robot"]["start"]
        s_init = CostStateAckermannCarFirstOrder(x_init, y_init, theta_init, 0)
        (x_goal, y_goal, theta_goal) = scenario_conf["robot"]["goal"]
        s_goal = StateAckermannCarFirstOrder(x_goal, y_goal, theta_goal)

        # planner = KDRRTPlanner(scenario_config_mod, workspace, obstacles,
        #                        q_start, carControlSpace, goal_bias=args.goal_bias, t_min=args.t_min_mcp,
        #                        t_max=args.t_max_mcp, k=args.n_mcp, goal_tolerance_pos=args.goal_tolerance_pos,
        #                        goal_tolerance_theta=args.goal_tolerance_theta, map_config_path=args.map_path,
        #                        robot_config_path=args.robot_config)

        kdrrt_planner = RRTPlanner(
            dynamics=dynamics,
            map2d=map2d,
            collision_checker=collision_checker,
            s_init=s_init,
            propagation_t_min=args.t_min_mcp,
            propagation_t_max=args.t_max_mcp,
            goal_tolerance_pos=args.goal_tolerance_pos,
            goal_tolerance_theta=args.goal_tolerance_theta,
        )

        driver = AckermannCarFirstOrderDriver(
            config_path=args.robot_config,
            model_path=args.model_path,
            model_path_onnx=args.model_path_onnx,
            dynamics=dynamics
        )

        if "rl" in args.candidates or "all" in args.candidates:
            if args.model_path is None:
                raise ValueError

            with console.status(
                    f"[bold green] RL round [{i}/{args.n}] [bold white] [{s_init.vector()}->{s_goal.vector()}] [bold "
                    f"grey] <{experiment_sub_dir}>") as status:
                t_start = time.perf_counter()
                if args.optimal:
                    (actions, states) = kdrrt_planner.find_optimal_trajectory(50, 0.1, s_goal, ExtensionModes.RL, driver,
                                                                         max_iterations=args.max_iterations)
                else:
                    (actions, states) = kdrrt_planner.find_trajectory_to(s_goal, ExtensionModes.RL, driver,
                                                                         max_iterations=args.max_iterations)
                t_end = time.perf_counter()
                t_diff = t_end - t_start
                success = len(actions) > 0
                write_results_to_result_file(result_path, t_diff, "rl", len(kdrrt_planner.nn_struct.nodes), success)
                write_traj_to_file(experiment_sub_dir, states, actions, "rl", 0.1)

            if args.store_traj_plot:
                if states:
                    print("Found trajectory! Visualizing!")
                    visualizer = visualize.Animation(args.map_path, q_start_overwrite=s_init, q_goal_overwrite=s_goal)
                    visualizer.add_trajectory(states)
                    tree_dir = os.path.join(experiment_sub_dir, "traj_plots")
                    os.makedirs(tree_dir, exist_ok=True)
                    tree_path = os.path.join(tree_dir, f"rl_traj_{time.time()}.pdf")
                    # visualizer.save_plot(tree_path)
                    visualizer.add_tree(kdrrt_planner.nn_struct.nodes)
                    plt.show()
                    visualizer.save_plot(tree_path)

        if "all" in args.candidates or "mcp" in args.candidates:
            with console.status(
                    f"[bold green] MCP round [{i}/{args.n}] [bold white] [{s_init.vector()}->{s_goal.vector()}] [bold "
                    f"grey] <{experiment_sub_dir}>") as status:
                t_start = time.perf_counter()
                if args.optimal:
                    (actions, states) = kdrrt_planner.find_optimal_trajectory(50, 50, s_goal, ExtensionModes.MCP, driver,
                                                                              max_iterations=args.max_iterations)
                else:
                    (actions, states) = kdrrt_planner.find_trajectory_to(s_goal, ExtensionModes.MCP, driver,
                                                                         max_iterations=args.max_iterations)
                t_end = time.perf_counter()
                t_diff = t_end - t_start
                success = len(actions) > 0
                write_results_to_result_file(result_path, t_diff, "rl", len(kdrrt_planner.nn_struct.nodes), success)
                write_traj_to_file(experiment_sub_dir, states, actions, "rl", 0.1)

            if args.store_traj_plot:
                if states:
                    print("Found trajectory! Visualizing!")
                    visualizer = visualize.Animation(args.map_path, q_start_overwrite=s_init, q_goal_overwrite=s_goal)
                    visualizer.add_trajectory(states)
                    tree_dir = os.path.join(experiment_sub_dir, "traj_plots")
                    os.makedirs(tree_dir, exist_ok=True)
                    tree_path = os.path.join(tree_dir, f"rl_traj_{time.time()}.pdf")
                    # visualizer.save_plot(tree_path)
                    visualizer.add_tree(kdrrt_planner.nn_struct.nodes)
                    plt.show()
                    visualizer.save_plot(tree_path)

        if "rl-onnx" in args.candidates or "all" in args.candidates:
            if args.model_path is None:
                raise ValueError

            with console.status(
                    f"[bold green] RL (ONNX) round [{i}/{args.n}] [bold white] [{s_init.vector()}->{s_goal.vector()}] [bold "
                    f"grey] <{experiment_sub_dir}>") as status:
                t_start = time.perf_counter()
                if args.optimal:
                    (actions, states) = kdrrt_planner.find_optimal_trajectory(50, 0.1, s_goal, ExtensionModes.RL_ONNX, driver,
                                                                              max_iterations=args.max_iterations)
                else:
                    (actions, states) = kdrrt_planner.find_trajectory_to(s_goal, ExtensionModes.RL_ONNX, driver,
                                                                         max_iterations=args.max_iterations)
                t_end = time.perf_counter()
                t_diff = t_end - t_start
                success = len(actions) > 0
                write_results_to_result_file(result_path, t_diff, "rl", len(kdrrt_planner.nn_struct.nodes), success)
                write_traj_to_file(experiment_sub_dir, states, actions, "rl", 0.1)

            if args.store_traj_plot:
                if states:
                    print("Found trajectory! Visualizing!")
                    visualizer = visualize.Animation(args.map_path, q_start_overwrite=s_init, q_goal_overwrite=s_goal)
                    visualizer.add_trajectory(states)
                    tree_dir = os.path.join(experiment_sub_dir, "traj_plots")
                    os.makedirs(tree_dir, exist_ok=True)
                    tree_path = os.path.join(tree_dir, f"rl_traj_{time.time()}.pdf")
                    # visualizer.save_plot(tree_path)
                    visualizer.add_tree(kdrrt_planner.nn_struct.nodes)
                    plt.show()
                    visualizer.save_plot(tree_path)

        # if "mcp" in args.candidates or "all" in args.candidates:
        #     import mcp
        #
        #     driver = mcp.MCPDriver(scenario_config_mod, args.robot_config)
        #     with console.status(
        #             f"[bold blue] GMCP round [{i}/{args.n}] [bold white] [{q_start}->{q_goal}] [bold grey] <{experiment_sub_dir}>") as status:
        #         planner._reset()
        #         t_start = time.perf_counter()
        #         (trajectory, actions) = planner.find_trajectory_to(
        #             q_goal, driver.propagate, args.max_iterations, rl=False, robot_config=args.robot_config)
        #         t_end = time.perf_counter()
        #         t_diff = t_end - t_start
        #         success = len(trajectory) > 0
        #         write_results_to_result_file(result_path, t_diff, "gmcp", planner.nn_struct.size(), success)
        #         write_traj_to_file(experiment_sub_dir, trajectory, actions, "gmcp", 0.1)
        #
        #     if args.store_traj_plot:
        #         visualizer = visualize.Animation(args.map_path, q_start_overwrite=q_start, q_goal_overwrite=q_goal)
        #
        #         visualizer.add_tree(planner.nn_struct.l)
        #
        #         if trajectory != []:
        #             visualizer.add_trajectory(trajectory)
        #             tree_dir = os.path.join(experiment_sub_dir, "traj_plots")
        #             os.makedirs(tree_dir, exist_ok=True)
        #             tree_path = os.path.join(tree_dir, f"mcp_traj_{time.time()}.pdf")
        #             visualizer.save_plot(tree_path)
