import argparse
import time
import os
from project_utils.utils import map_to_diff_range
import yaml
from rich.logging import RichHandler
import logging.config
import importlib
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
import logging

plt.style.use('dark_background')


def plot_rewards(rewards):
    xs = [i for i in range(len(rewards))]

    cumm_rewards = []
    cumm_reward = 0
    for i in range(len(rewards)):
        cumm_reward += rewards[i]
        cumm_rewards.append(cumm_reward)

    (fig, axs) = plt.subplots(2, 1, sharex="all")
    axs[0].plot(xs, rewards)
    axs[0].set_ylabel("$r_j$")
    axs[0].set_title("reward")
    axs[1].plot(xs, cumm_rewards)
    axs[1].set_title("cummulative reward")
    axs[1].set_ylabel("$\sum^{j}_{i=0} r_{i}$")
    axs[1].set_xlabel("step")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_module", type=str,
                        help="specifies the module where the environment (CustomEnv) is defined")
    parser.add_argument("--manual", action="store_true",
                        help="if this flag is set, the user can initiate each action by pressing enter")
    parser.add_argument("--arg1", type=float, required=False,
                        help="(float) fix the first argument of the action (else: random)")
    parser.add_argument("--arg2", type=float, required=False,
                        help="(float) fix the second argument of the action (else: random)")
    parser.add_argument("--trace_mode", action="store_true")
    parser.add_argument("--no_rendering", action="store_true",
                        help="if set, the env will not be rendered. can be used when generating trajectories")
    parser.add_argument("--config", required=False)
    parser.add_argument("--trajectory_dir", type=str, required=False,
                        help="if set, the script will save the models trajectory to a new file in the given directory")
    parser.add_argument("--dont_convert_actions", dest="convert_actions", action="store_false")
    parser.add_argument("--plot_rewards", action="store_true", help="plot the reward function output")
    parser.add_argument("-n", type=int, default=-1)
    args = parser.parse_args()

    env_mod = importlib.import_module(args.env_module)

    env = env_mod.CustomEnv(trace_mode=args.trace_mode, config_path=args.config)
    check_env(env)
    env.reset()

    if args.trajectory_dir is not None:
        os.makedirs(args.trajectory_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config, "r"))

    logging.config.fileConfig("rl/log.ini")
    logger = logging.getLogger(__file__)
    logger.addHandler(RichHandler())

    rew_sum = 0
    n = args.n
    action_count = 0
    states = []
    actions = []

    rewards = []

    while True:
        if not args.no_rendering:
            env.render()

        if args.manual:
            input("Press to perform a random action.")

        action = env.action_space.sample()

        if args.convert_actions:
            version_id = env.version_id
            converted_action_0, converted_action_1 = env.dynamics.convert_normalized_action(action)
            actions.append([float(converted_action_0), float(converted_action_1)])
        else:
            actions.append([float(item) for item in action])

        states.append([float(item) for item in env.get_state()])

        if args.arg1 is not None:
            action[0] = args.arg1
        if args.arg2 is not None and len(action) > 1:
            action[1] = args.arg2

        logger.info(f"action: {action}")
        (observation, reward, terminated, _) = env.step(action)
        action_count += 1
        rew_sum += reward
        rewards.append(reward)
        logger.debug(f"observation, reward, terminated: {observation} {reward} {terminated} {action_count}")
        logger.debug(f"internal action count: {env.action_count}")
        if terminated:
            logger.info(f"TERMINATED! Resetting")
            logger.info(f"+++++ REWARD SUM: {rew_sum} ++++ ")
            logger.info(f"++++ actions: {action_count} ++++ ")

            if args.plot_rewards:
                plot_rewards(rewards)

            if args.trajectory_dir is not None:
                traj_file_path = os.path.join(args.trajectory_dir,
                                              f"trajectory_{args.config.split('/')[-1].split('.')[0]}_{time.time()}.yaml")
                with open(traj_file_path, "w") as f:
                    output_dict = {
                        "actions": actions,
                        "states": states
                    }
                    yaml.safe_dump(output_dict, f)

            rew_sum = 0
            rewards = []
            action_count = 0
            env.reset()
            actions = []
            states = []
            if n > 0:
                n -= 1
            if n == 0:
                break
