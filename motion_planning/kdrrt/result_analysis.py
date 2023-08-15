import pandas as pd
import math
import yaml
import os
from matplotlib import pyplot as plt
import numpy as np
from argparse import ArgumentParser
plt.style.use('fivethirtyeight')

"""
tool for plotting the experiment results
"""

def plot_with_mean(ax, df, ylabel, column_selector):
    """
    plot the data for rl and comparison as a scatter plot and hline (mean)
    args:
    - ax: axis on which to plot
    - df: data frame with the source data
    - column_selector: name of the column which should be plotted
    """
    rl_df = df[df['approach']=="rl"].reset_index()
    mcp_df = df[df['approach']=="gmcp"].reset_index()

    rl_df_slice = rl_df[column_selector]
    mcp_df_slice = mcp_df[column_selector]

    rl_mean = rl_df_slice.mean()
    mcp_mean = mcp_df_slice.mean()


    xticks = [x for x in range(len(rl_df_slice))]
    ax.scatter(xticks, rl_df_slice, marker = "x", label = "rl", color="#340089")
    ax.hlines([rl_mean], label = "mean (rl)", xmin = 0, xmax = len(rl_df)-1, linestyle="--", color="#e11a5f")

    xticks = [x for x in range(len(mcp_df_slice))]
    ax.scatter(xticks, mcp_df_slice, marker = "o", label = "mcp", color="#ffc75f")
    ax.hlines([mcp_mean], label = "mean (mcp)", xmin = 0, xmax = len(rl_df)-1, linestyle="--", color="#ff9671")
    ax.set_ylabel(ylabel)
    ax.set_xticks(xticks)
    ax.legend()

def plot_with_mean_complete(axs, df):
    """
    plot t_diff and nr_of_nodes from the given dataframe
    - axs: axis of the plot (needs to be at least len 2)
    - df: data frame with the source data
    """
    if len(axs) < 2:
        raise ValueError
    plot_with_mean(axs[0], df, "$\Delta$ t [sec]", "t_diff")
    plot_with_mean(axs[1], df, "# nodes", "nr_of_nodes")

def extract_trajectory_meta(traj_file_path:str):
    file_name = traj_file_path.split("/")[-1]
    parts = file_name.split("_")
    t_str = parts[0]
    algo_str = parts[1].split(".")[0]
    traj_dict = yaml.safe_load(open(traj_file_path, "r"))
    algo = traj_dict["algo"]
    trajectory = traj_dict["traj"]
    controls = traj_dict["controls"]
    dt = traj_dict["dt"]
    total_time = dt * len(controls)
    path_length = 0
    last_c = trajectory[0]
    print(f"last_c: {last_c}")
    if len(trajectory) > 1:
        for c in trajectory[1:]:
            d = math.dist((last_c[0], last_c[1]), (c[0], c[1]))
            path_length += d
            last_c = c
    print(f"extraction result for {traj_file_path}: l: {path_length} t: {total_time}")
    print(f"len(traj): {len(trajectory)} len(conrols): {len(controls)}")
    return (path_length, total_time)

def extract_traj_dir_meta(traj_dict_path:str):
    traj_files_rl = [os.path.join(traj_dict_path, traj_file) for traj_file in os.listdir(traj_dict_path) if traj_file.endswith("rl.yaml")]
    traj_files_mcp= [os.path.join(traj_dict_path, traj_file) for traj_file in os.listdir(traj_dict_path) if traj_file.endswith("gmcp.yaml")]
    print("file arrays done")
    rl_traj_meta = []
    mcp_traj_meta = []
    for tf in traj_files_rl:
        rl_traj_meta.append(extract_trajectory_meta(tf))
    print("rl_traj_meta done")
    for tf in traj_files_mcp:
        mcp_traj_meta.append(extract_trajectory_meta(tf))
    print("mcp_traj_meta done")
    
    return (rl_traj_meta, mcp_traj_meta)
    
def plot_traj_meta_comparison(rl_traj_meta, mcp_traj_meta):
    print("plotting now")
    (fig, axs) = plt.subplots(2,1, sharex=True, figsize=(20,20))
    rl_lengths = [tm[0] for tm in rl_traj_meta]
    rl_times = [tm[1] for tm in rl_traj_meta]
    mcp_lengths = [tm[0] for tm in mcp_traj_meta]
    mcp_times = [tm[1] for tm in mcp_traj_meta]
    indexes_rl = list(range(len(rl_lengths)))
    indexes_mcp = list(range(len(mcp_lengths)))
    axs[0].scatter(indexes_rl, rl_times, marker="x", label="rl")
    axs[0].scatter(indexes_mcp, mcp_times, marker="o", label="mcp")
    axs[0].legend()
    axs[0].set_ylabel("path execution time [sec]")
    axs[1].scatter(indexes_rl, rl_lengths, marker="x", label="rl")
    axs[1].scatter(indexes_mcp, mcp_lengths, marker="o", label="mcp")
    axs[1].legend()
    axs[1].set_ylabel("path length [m]")
    axs[1].set_xlabel("trial id")
    axs[1].set_xticks(indexes_mcp)
    print("done")
    return (fig, axs)

def plot_traj_experiment_results(experiment_sub_dir, plot_dir):
    traj_dir = os.path.join(experiment_sub_dir, "trajectories")
    (rl_traj_meta, mcp_traj_meta) = extract_traj_dir_meta(traj_dir)
    print("extraction done")
    (fig, axs) = plot_traj_meta_comparison(rl_traj_meta, mcp_traj_meta)
    fig.savefig(os.path.join(plot_dir, "trajectory_eval.jpg"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("eval_dir", help="directory where the experiment results are stored", type=str)
    parser.add_argument("--plot_dir", required=False, help="directory where the generated plots should be stored. If no path is specified, the evaluation dir is selected", default="plots")
    args = parser.parse_args()

    eval_dir = args.eval_dir
    result_path = os.path.join(eval_dir, "results.csv")
    if args.plot_dir is not None:
        plot_dir = os.path.join(eval_dir, args.plot_dir)
        print(f"plot dir: {plot_dir}")
        os.makedirs(plot_dir, exist_ok=True)
        output_path = os.path.join(plot_dir, "plot.jpg")
    else:
        output_path = os.path.join(eval_dir, "plot.jpg")

    df = pd.read_csv(result_path, header=0,parse_dates=True)

    (fig, axs) = plt.subplots(2,1, sharex=True, figsize=(20,20))
    plot_with_mean_complete(axs, df)
    axs[-1].set_xlabel("trial id")
    plt.savefig(output_path)
    print("first plot done")
    plot_traj_experiment_results(eval_dir, plot_dir = plot_dir)

    plt.show()

