import yaml
from rich.console import Console
from rich.progress import track
import transformations
import os
import math


def state_obeys_boundaries(state, x_bounds = (-1,1), y_bounds = (-1,1), lin_vel_bounds = (-0.1, 0.5), phi_bounds = (-math.pi/3, math.pi/3)):

    (x,y,theta,lin_vel,phi,_,_) = state

    if not(x_bounds[0] <= x <= x_bounds[1]):
        return False
    if not(y_bounds[0] <= y <= y_bounds[1]):
        return False
    if not(lin_vel_bounds[0] <= lin_vel <= lin_vel_bounds[1]):
        return False
    if not(phi_bounds[0] <= phi <= phi_bounds[1]):
        return False
    return True



CONFIG_PATH = "config.yaml"
RAW_TRAJ_DIR = "./raw"
CONVERTED_TRAJ_DIR = "./converted"
GOLD_TRAJ_DIR = "./gold"
golden_traj_nr_target = 1000

console = Console()


os.makedirs(CONVERTED_TRAJ_DIR, exist_ok=True)
os.makedirs(GOLD_TRAJ_DIR, exist_ok=True)
golden_trajs = 0
cnt = 0
file_names = os.listdir(RAW_TRAJ_DIR)

for traj_name in track(file_names, "Transforming and filtering ..."):
    golden = True
    if not traj_name.endswith(".yaml"):
        continue
    traj_path = os.path.join(RAW_TRAJ_DIR, traj_name)
    traj = yaml.safe_load(open(traj_path, "r"))
    states = traj['states']
    actions = traj['actions']
    start_state = states[0]
    (x_t, y_t, theta_t, lin_vel_target, phi_target, _, _) = states[-1]
    transformed_states = []

    for (x, y, theta, lin_vel, phi, _, _) in states:
        (x_n, y_n, theta_n) = transformations.transform_to_target_frame(x,y,theta,x_t, y_t, theta_t)
        transformed_state = (x_n, y_n, theta_n, lin_vel, phi, lin_vel_target, phi_target)
        golden = state_obeys_boundaries(transformed_state)
        transformed_states.append(transformed_state)
    
    out_traj_path = os.path.join(CONVERTED_TRAJ_DIR, traj_name)
    yaml.safe_dump({
        "states": transformed_states,
        "actions": actions
    }, open(out_traj_path, "w"))
    cnt += 1
    if golden and golden_trajs < golden_traj_nr_target:
        golden_traj_out_path = os.path.join(GOLD_TRAJ_DIR,traj_name)
        yaml.safe_dump({
            "states": transformed_states,
            "actions": actions
        }, open(golden_traj_out_path, "w"))
        golden_trajs += 1
console.print(f"[green]>Conversion finished<")
console.print(f">total trajectories : [white]{cnt}")
console.print(f">golden trajectories: [green]{golden_trajs}")
