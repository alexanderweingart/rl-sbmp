import random
import numpy as np
import yaml
import time
import matplotlib.pyplot as plt
import math
import numpy as np
import argparse


def sample_in_range(min_val, max_val):
    return (max_val-min_val) * random.random() + min_val


def plot_sample_poses(starting_poses):
    xs = [starting_pose[0] for starting_pose in starting_poses]
    ys = [starting_pose[1] for starting_pose in starting_poses]
    thetas = [starting_pose[2] for starting_pose in starting_poses]
    (fig, axs) = plt.subplots(2, 1)
    axs[0].scatter(xs, ys)
    axs[0].set_xticks([x/10-1 for x in range(20)])
    axs[0].set_yticks([y/10-1 for y in range(20)])
    axs[1].scatter([i for i in range(len(thetas))], thetas)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config",
                        help="path to the experiment config")
    parser.add_argument("-n", help="nr of samples", default=20, type=int)
    parser.add_argument("-o", "--out_path", help="output path",
                        default="starting_config.yaml", type=str)
    parser.add_argument(
        "--plot_samples", help="if set, matplotlib will be used to visualise the samples")
    parser.add_argument("--accl", action="store_true",
                        help="generate stating configs that are suitable for the accl. car as well (x, y, theta, lin_vel, phi, lin_vel_target, phi_target)")

    parser.add_argument("--lin_vel_min", type=float,
                        help="min linear velocity", default=-0.1)
    parser.add_argument("--lin_vel_max", default=0.5,
                        help="max linear velocity")
    parser.add_argument("--phi_min", default=-np.pi/3,
                        help="min phi (steering angle)")
    parser.add_argument("--phi_max", default=np.pi/3,
                        help="max phi (steering angle)")

    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))

    config_keys = config.keys()
    arg_keys = args.__dict__.keys()
    for key in config_keys:
        if key in arg_keys:
            args.__dict__[key] = config[key]

    for key in arg_keys:
        print("setup:")
        print(f"{key}: {args.__dict__[key]}")

    (theta_min, theta_max) = (0, 2 * np.pi)

    (r_min, r_max) = (0, 0.5)
    (polar_phi_min, polar_phi_max) = (0, 2*np.pi)

    rs = [sample_in_range(r_min, r_max) for _ in range(args.n)]
    polar_phis = [sample_in_range(polar_phi_min, polar_phi_max)
                  for _ in range(args.n)]

    polar_samples = zip(rs, polar_phis)
    if args.accl:
        start_configs = [
            (math.cos(polar_phi) * r,  # x
             math.sin(polar_phi) * r,  # y
             sample_in_range(theta_min, theta_max),  # theta
             sample_in_range(args.lin_vel_min, args.lin_vel_max),  # lin_vel
             sample_in_range(args.phi_min, args.phi_max),  # phi
             # lin_vel_target
             sample_in_range(args.lin_vel_min, args.lin_vel_max),
             sample_in_range(args.phi_min, args.phi_max))  # phi_target
            for (r, polar_phi) in polar_samples
        ]

    else:
        start_configs = [(math.cos(polar_phi)*r, math.sin(polar_phi)*r, sample_in_range(
            theta_min, theta_max)) for (r, polar_phi) in polar_samples]

    if (args.plot_samples):
        plot_sample_poses(start_configs)

    output_dict = {
        "start_configs": start_configs
    }

    with open(args.out_path, "w") as f:
        yaml.safe_dump(output_dict, f)
