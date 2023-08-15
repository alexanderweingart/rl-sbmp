import yaml
from rich.progress import track
import os
import time
import importlib
import argparse

def render_trajectory(env, path):
    trajectory = yaml.safe_load(open(path, "r"))
    states = trajectory['states']
    sleep_time = 1/args.fr
    t_temp = time.perf_counter()
    for (x, y, theta, lin_vel, phi, lin_vel_target, phi_target) in states:
        env.set_state(x,y,theta,lin_vel,phi,lin_vel_target,phi_target)
        env.render()
        t_diff = sleep_time - (time.perf_counter() - t_temp)
        if t_diff > 0:
            time.sleep(t_diff)
        t_temp = time.perf_counter()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env", help="path to environment module")
    parser.add_argument("traj_dir", help="path to trajectory dir")
    parser.add_argument("-fr", help="frame rate (in Hz)", default=10, type=int)
    args = parser.parse_args()


    CONFIG_PATH = "config.yaml"

    env_module = importlib.import_module(args.env)
    env = env_module.CustomEnv(config_path=CONFIG_PATH)

    for fn in track(os.listdir(args.traj_dir)):
        if not fn.startswith("trajectory_"):
            continue
        f_path = os.path.join(args.traj_dir, fn)
        render_trajectory(env, f_path)
    

        














