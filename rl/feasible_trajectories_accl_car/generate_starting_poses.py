import yaml
from rich.console import Console
import os

TRAJ_DIR = "./gold"

# POSE_FILE = "./99_starting_poses_20_steps.yaml"
POSE_FILE = "./1000_starting_poses_20_steps.yaml"

console = Console()

starting_states = []
for traj_name in os.listdir(TRAJ_DIR):
    if not traj_name.endswith(".yaml"):
        continue
    traj_path = os.path.join(TRAJ_DIR, traj_name)
    traj = yaml.safe_load(open(traj_path, "r"))
    first_state = traj['states'][0]
    starting_states.append(first_state)
yaml.safe_dump({
    "training_set": starting_states
}, open(POSE_FILE, "w"))
console.print("[green]FINISHED")
console.print(
    f"wrote {len(starting_states)} starting positions to [bold] {POSE_FILE}")
