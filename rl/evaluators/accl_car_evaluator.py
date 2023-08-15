from rl.environments import accl_car
import math
import yaml
from stable_baselines3 import PPO, A2C, SAC
from argparse import ArgumentParser
from rich.progress import track
from rich import print
import matplotlib.pyplot as plt
import numpy as np
# plt.style.use('dark_background')
plt.style.use('seaborn-v0_8-darkgrid')

def plot_state_history(states, config):
    # (x,y,theta,lin_vel,phi,lin_vel_target,phi_target)
    states_np = np.array(states)
    lin_vel_min = config['lin_vel_min']
    lin_vel_max = config['lin_vel_max']
    lin_vel_target = states[0][5]

    xs = [i for i in range(len(states))]
    fig, axs = plt.subplots(1,1,sharex=True) 
    axs = [axs]
    axs[0].plot(xs, states_np[:, 3])
    axs[0].hlines([lin_vel_min, lin_vel_max], colors=['red'], linestyles=['dashed'], label="lin. vel. boundaries", xmin=0, xmax = len(xs)-1)
    axs[0].hlines([lin_vel_target], colors=['green'], label="lin. vel. target", xmin=0, xmax = len(xs)-1)
    axs[0].set_title("linear velocity")
    axs[0].set_ylabel("$v \ [\frac{m}{s}]$")
    axs[0].set_xlabel("$t_i \ [0.1 \ s]$")
    plt.show()
    

def load_model(algo_str, model_path, env):
    if algo_str == "PPO":
        return PPO.load(model_path, env)
    if algo_str == "A2C":
        return A2C.load(model_path, env)
    if algo_str == "SAC":
        return SAC.load(model_path, env)
    
    raise ValueError(f"selected algorithm is not supported [{algo_str}]")


def estimate_path_length(path):
    """
    Returns the estimated length of the given path
    The estimation is based on the euclidean distance between the states
    """
    if len(path) < 2:
        raise ValueError("Path needs to have at least two states")

    path_length = 0
    print(f"len(path): {len(path)}")
    for i in range(len(path)-2):
        print(f"i: {i}")
        path_length += math.dist(path[i][:2], path[i+1][:2])

    return path_length

    


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", help="path to the models config", type=str)
    parser.add_argument("--model", "-m", help="path to the model", type=str)
    parser.add_argument("--traj_path", "-t", help="path to the trajectory file")
    parser.add_argument("--algo", default="PPO", help="rl-algorithm used to train the given model. default: PPO")
    parser.add_argument("--print_round_analysis","-ra", action="store_true")
    args = parser.parse_args()


    if args.config is not None: # config params overwrite command line args
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)
            for key in config.keys():
                args.__dict__[key] = config[key]

    if args.traj_path is not None:
        trajectories = yaml.safe_load(open(args.traj_path, "r"))


    env = accl_car.CustomEnv(config_path = args.config, trace = True)

    model = load_model(args.algo, args.model, env)
    first = True

    for traj in track(trajectories):
        start_config = traj['start']
        goal_config = traj['goal']

        (x_0, y_0, theta_0, lin_vel_0, phi_0) = start_config
        (_, _, _, lin_vel_goal, phi_goal) = goal_config

        obs = env.set_state(x_0, y_0, theta_0, lin_vel_0, phi_0, lin_vel_goal, phi_goal)


        optimizer_path = [(x,y,theta) for (x,y,theta,_,_) in traj['states']]


        done = False

        model_states = []

        while not done:
            env.render(perfect_path = optimizer_path)
            if first:
                input("Press enter to start")
                first = False
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            model_states.append(env._state())
        
        if args.print_round_analysis:
            model_path = [(x,y,theta) for (x,y,theta,_,_,_,_) in model_states]

            path_length_optimizer = estimate_path_length(optimizer_path) 
            path_length_model = estimate_path_length(model_path)

            print(f"[green bold]> PATH ANALYSIS <")
            print(f"> path optimizer: ")
            print(optimizer_path)
            print(f"> path model:")
            print(model_path)

            print(f"> optimizer: {path_length_optimizer:.3f}")
            print(f"> model:     {path_length_model:.3f}")
            plot_state_history(model_states, config)

            print("-"*20)
            input("")
    
        
        
        
            




