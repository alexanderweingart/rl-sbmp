import time
import importlib
import yaml
import os
from rl.onnx_helper.onnx_wrapper import OnnxModelWrapper

from stable_baselines3 import PPO, A2C, SAC
import argparse

"""
This script can be used to load trained models and watch them perform in their simulated environment.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_module", type=str)
    parser.add_argument("--onnx", action="store_true")
    parser.add_argument("--seed", required=False, type=int)
    parser.add_argument("--algo", default="ppo", choices=["ppo", "a2c", "sac"], type=str)
    parser.add_argument("--vel", required=False, type=float)
    parser.add_argument("--path", required=True, type=str)
    parser.add_argument("--start_pose_file", required=False, type=str)
    parser.add_argument("--radius", default=0.5)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--config", required=False, type=str)
    parser.add_argument("-n", default=10, type=int)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--step_through", action="store_true")
    parser.add_argument("--show_perfect", action="store_true", help="visualise the perfect path (for the case of "
                                                                    "fixed lin velocity)")
    parser.add_argument("--shot_dir", type=str, required=False)
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--result_trajectory_dir", "-rtd", required=False)
    parser.add_argument("--no_render", action="store_true")
    args = parser.parse_args()

    # load the config parameters and store them into the args dictionary
    # this overwrites the parameters set by the command line arguments! (design decision)

    if args.config is not None:  # config params overwrite command line args
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)
            for key in config.keys():
                args.__dict__[key] = config[key]

    if args.show_perfect:
        try:
            import dubins
        except ModuleNotFoundError:
            print("Rendering the perfect (Dubin's) Path needs the dubins module.")
            print("install the module and try again.")

    env_mod = importlib.import_module(args.env_module)

    env = env_mod.CustomEnv(config_path=args.config, trace_mode=args.trace)


    start_poses = []
    # load start poses from yaml file. can be used for direct comparison or to estimate the performance
    # of the model on its training-set
    if args.start_pose_file is not None:
        with open(args.start_pose_file, "r") as f:
            pose_file = yaml.safe_load(f)
            for pose in pose_file['training_set']:
                start_poses.append(pose)

    if not os.path.exists(args.path):
        print(f"There is no model at [{args.path}]")
        exit(2)

    if args.shot_dir is not None:
        if not os.path.exists(args.shot_dir):
            choice = input(f"There is no dir at {args.shot_dir}. Should I create it? [Y|n]")
            if choice != "n":
                os.makedirs(args.shot_dir, exist_ok=True)

    if args.algo == "ppo":
        if args.onnx:
            model = OnnxModelWrapper(env, onnx_path=args.path)
        else:
            model = PPO.load(args.path, env)

    elif args.algo == "a2c":
        model = A2C.load(args.path, env)
    elif args.algo == "sac":
        model = SAC.load(args.path, env)
    else:
        raise ValueError

    success = 0
    reward_sums = []

    stats = []
    sample_random = False

    for i in range(args.n):
        actions = []
        states = []
        if not sample_random:
            if i < len(start_poses):
                obs = env.set_state(*start_poses[i])
            else:
                sample_random = True
                obs = env.reset()
                state = env.get_state()
                start_poses.append(list(obs))
        else:
            obs = env.reset()
            start_poses.append(list(obs))

        perfect_path = []
        if args.show_perfect:  # visualize the perfect path
            x, y, theta = (env.x, env.y, env.theta)
            v = env.max_lin_vel
            omega = env.max_ang_vel
            turning_radius = v / omega
            path = dubins.shortest_path((x, y, theta), (0, 0, 0), turning_radius)
            perfect_path, _ = path.sample_many(v)

        done = False
        reward_sum = 0
        states.append([float(item) for item in env.get_state()])
        while not done:
            if not args.no_render:
                env.render(perfect_path=perfect_path)

            action, _states = model.predict(obs, deterministic=True)

            if args.result_trajectory_dir is not None:
                converted_action_0, converted_action_1 = env.dynamics.convert_normalized_action(action)
                actions.append([float(converted_action_0), float(converted_action_1)])

            if args.step_through:
                print(f"Models action choice: {action}")
                input("press enter to act on it")


            obs, rewards, done, info = env.step(action)



            states.append([float(item) for item in env.get_state()])

            if args.step_through:
                print(f"obs: {obs}")
            reward_sum += rewards
            if done:
                print(f"TERMINATED: reward: {rewards} [{reward_sum}]")
                if rewards > 0:
                    success += 1
                reward_sums.append(reward_sum)
                if args.show_perfect:
                    if len(states) > 0:
                        first_state = states[0]
                        stats.append((first_state[0], first_state[1], first_state[2],
                                      len(perfect_path), len(env.trace)))

                if args.result_trajectory_dir is not None:
                    os.makedirs(args.result_trajectory_dir, exist_ok=True)
                    config_name = args.config.split("/")[-1].split(".")[0]

                    trajectory_file = os.path.join(args.result_trajectory_dir,
                                                   f"trajectory_{config_name}_{time.time()}.yaml")
                    trajectory_dict = {
                        'actions': actions,
                        'states': states
                    }
                    yaml.safe_dump(trajectory_dict, open(trajectory_file, "a"))

                if args.shot_dir:
                    env.render(perfect_path=perfect_path, screenshot_dir=args.shot_dir)

                break
            time.sleep(0.01)

    print(f"result: {success} / {args.n}")
    print(f"reward sums: {reward_sums}")
    print(f"average: {sum(reward_sums) / len(reward_sums)}")
    with open("stats.csv", "a") as f:
        f.write("x,y,theta,perfect,result\n")
        for (x, y, theta, perfect, result) in stats:
            f.write(f"{x:.3f},{y:.3f},{theta:.3f},{perfect},{result}\n")
