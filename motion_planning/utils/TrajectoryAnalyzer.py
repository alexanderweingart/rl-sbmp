import argparse
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import yaml
from spaces.State import StateAckermannCarFirstOrder
from rich.progress import track
from rich import print

from motion_planning.kdrrt.visualize import Animation

sb.set_style("whitegrid")

def construct_dataframe_for_experiment_run_dir(run_dir_path, runtime_sec_if_failed=240):
    run_dirs = [p for p in os.listdir(run_dir_path) if os.path.isdir(os.path.join(run_dir_path, p))]
    if len(run_dirs) == 0:
        raise Exception("The given input directory does not include any directories.")

    costs = []
    ts_start = []
    ts_end = []
    run_ids = []
    ts_start_run = []

    for j, run_dir in enumerate(run_dirs):
        print(f"[blue]reading dir {run_dir} ({j+1}/{len(run_dirs)})")
        run_id = run_dir
        path = os.path.join(run_dir_path, run_dir)
        trajs = [fn for fn in os.listdir(path) if "intermediate_trajectory" in fn and
                 not os.path.isdir(os.path.join(path, fn)) and fn.endswith(".yaml")]
        trajs.sort()

        t_start_run = None
        empty_trajs = 0
        for i, traj in enumerate(trajs):

            traj_path = os.path.join(path, traj)
            traj_dict = yaml.safe_load(open(traj_path, "r"))

            if traj_dict is None or 't_total_sec' not in traj_dict:
                print("[red] incomplete trajectory file. (this might be normal)")
                print(f"path: {traj_path}")
                print(f"[red] skipping")
                empty_trajs += 1
                continue

            costs.append(traj_dict['t_total_sec'])
            t_start_datetime = datetime.datetime.strptime(traj_dict['t_start'], "%Y-%m-%d %H:%M:%S.%f")

            if t_start_run is None or t_start_datetime < t_start_run:
                t_start_run = t_start_datetime

            ts_start.append(t_start_datetime)
            ts_end.append(datetime.datetime.strptime(traj_dict['t_finished'], "%Y-%m-%d %H:%M:%S.%f"))
            run_ids.append(j)

        if len(trajs) == 0 or len(trajs) == empty_trajs:
            print(f"[red] the run in {run_dir} didn't find any trajectory...")
            print(f"setting t_end_to inf")
            print(f"because there is no t_start, we will gather t_start from the directory name (+1s for startup dalay)")
            t_start = datetime.datetime.strptime(run_dir, "%Y-%m-%d_%H-%M-%S") + datetime.timedelta(seconds=1)
            costs.append(np.inf)
            run_ids.append(j)
            t_end = t_start + datetime.timedelta(seconds=runtime_sec_if_failed)
            ts_start.append(t_start)
            ts_end.append(t_end)
            ts_start_run.append(t_start)
        else:
            ts_start_run += [t_start_run for i in range(len(trajs)-empty_trajs)]

    df = pd.DataFrame(data={
        "run_id": run_ids,
        "t_start": ts_start,
        "t_start_run": ts_start_run,
        "t_end": ts_end,
        "cost": costs
    })

    first_t_start = list(df['t_start'].sort_values())[0]

    df['t_start_rel'] = df['t_start'] - first_t_start
    df['t_end_rel'] = df['t_end'] - first_t_start
    df['run_dur'] = df['t_end'] - df['t_start_run']
    return df


def extract_first_success_ts_rel(df):
    df_sorted_by_t_end_and_duplicates_dropped = df.sort_values(by="t_end").drop_duplicates(subset=["run_id"],
                                                                                           keep='first')

    t_end_rel = df_sorted_by_t_end_and_duplicates_dropped['run_dur']
    t_end_rel_secs = [temp.total_seconds() for temp in t_end_rel]

    return sorted(t_end_rel_secs)


def plot_costs(ax, df):
    sorted_df = df.sort_values(by="t_end_rel")
    data = []
    for run_id in df['run_id'].unique():
        run_id_df = sorted_df[sorted_df['run_id'] == run_id]
        costs = run_id_df['cost']
        ts = run_id_df['run_dur']
        ts_secs = [t.total_seconds() for t in ts]
        data += [(ts_secs[i], costs.to_list()[i]) for i in range(len(costs))]
        ax.scatter(ts_secs, costs, marker="x")
    arr = np.asarray(data)
    z = np.poly1d(np.polyfit(arr[:, 0], arr[:, 1], 2))
    x_min = np.min(arr[:, 0])
    x_max = np.max(arr[:, 0])
    x = np.linspace(x_min, x_max, 100)
    ax.plot(x, z(x), "--", color="r", label="polyfit")
    ax.set_ylabel("$\Delta t_{path}$")


def construct_metrics_dataframe(df, time_step_sec: float = 1):
    sorted_df = df.sort_values(by="run_dur")
    run_ids = df['run_id'].unique()
    t_end = sorted_df['run_dur'].tolist()[-1].total_seconds()
    t_start = sorted_df['run_dur'].tolist()[0].total_seconds()

    ys = []
    t = t_start
    meta_min_ys = []
    meta_max_ys = []
    median_start_i = 0
    standard_deviations = []
    success_percs = []

    # traverse the results from t_start to t_end in time_step_sec steps
    duration_secs = int(np.ceil((t_end - t_start)))
    ts = np.asarray([t_start + t for t in range(duration_secs)])
    run_id_mins = {}
    for run_id in run_ids:
        id_df = df[df['run_id'] == run_id]
        cost_sorted_id_df = id_df.sort_values(by="cost", ascending=False)
        min_arr = np.asarray([np.inf for i in range(len(ts))])
        for i, row in cost_sorted_id_df.iterrows():
            t = row['run_dur'].total_seconds()
            min_arr[ts >= t] = row['cost']
        run_id_mins[run_id] = min_arr

    for i, t in enumerate(ts):
        min_ys = []
        #  get the best result of every run id (up to the point t)
        for run_id in run_ids:
            min_y = run_id_mins[run_id][i]
            min_ys.append(min_y)
        min_ys = np.asarray(min_ys)
        success_perc = (np.count_nonzero(min_ys != np.inf) / len(run_ids))*100  # cost is set to inf if not returned yet
        min_ys_arr = np.asarray(min_ys)

        min_ys_arr = min_ys_arr[~np.isnan(min_ys_arr)]
        ys.append(np.median(min_ys_arr))
        standard_deviations.append(np.std(min_ys_arr))
        meta_min_ys.append(min_ys_arr.min())
        meta_max_ys.append(min_ys_arr.max())
        success_percs.append(success_perc)

    return_df = pd.DataFrame(data={"t": ts, "min_cost": meta_min_ys, "max_cost": meta_max_ys, "median_cost": ys,
                                   "std_cost": standard_deviations, "success_perc": success_percs})

    return return_df


def plot_cost_plot_v2(ax, df, metrics_df, median_start, time_step_sec: float = 1, label="", runtime=None, plot_range=False):
    sorted_df = df.sort_values(by="run_dur")
    run_ids = df['run_id'].unique()

    ts = np.asarray(metrics_df['t'])
    # plot all runs as a step function
    if len(run_ids) < 10:  # over 10, the plot gets ugly if there are too many individual runs
        for i, run_id in enumerate(run_ids):
            id_df = sorted_df[sorted_df['run_id'] == run_id]
            best_cost = id_df['cost'].tolist()[-1]
            id_ts = [run_dur.total_seconds() for run_dur in id_df['run_dur'].tolist()]

            xs =  id_ts + [ts[-1]]
            ys = id_df['cost'].tolist() + [best_cost]
            if runtime is not None and len(xs) > 0 and runtime > ts[-1]:
                xs += [runtime]
                ys += [best_cost]

            ax.step(xs, ys, where='post', color="grey", label="run" if i == len(run_ids) - 1 else None)

    # draw the range (between min and max val of each timestep
    xs = metrics_df['t'].tolist()
    ys_min = metrics_df['min_cost'].tolist()
    ys_max = metrics_df['max_cost'].tolist()
    if runtime is not None and len(xs) > 0 and runtime > xs[-1]:
        xs += [runtime]
        ys_min += [ys_min[-1]]
        ys_max += [ys_max[-1]]
    if plot_range:
        ax.fill_between(xs, ys_min, ys_max, step='post', color="grey",
                        alpha=0.2, label="range")

    rel_median_rows = metrics_df[metrics_df['t'] >= median_start]

    xs = rel_median_rows['t'].tolist()
    ys = rel_median_rows['median_cost'].tolist()

    if runtime is not None and len(xs) > 0 and runtime > xs[-1]:
        xs += [runtime]
        ys += [ys[-1]]

    # median
    median_plt = ax.step(xs, ys, where='post', label=f"median {label}")

    # std
    median_arr = np.asarray(rel_median_rows['median_cost'])
    std_arr = np.asarray(rel_median_rows['std_cost'])
    upper_std = median_arr + std_arr
    lower_std = median_arr - std_arr

    xs = rel_median_rows['t'].tolist()
    ys_min = lower_std.tolist()
    ys_max = upper_std.tolist()
    if runtime is not None and len(xs) > 0 and runtime > xs[-1]:
        xs += [runtime]
        ys_min += [ys_min[-1]]
        ys_max += [ys_max[-1]]
    ax.fill_between(xs, ys_min, ys_max, step='post', color=median_plt[0].get_color(), alpha=0.1, label="std")


def add_success_plot(ax, metric_df, label="", runtime=None):
    xs = metric_df['t'].tolist()
    ys = metric_df['success_perc'].tolist()
    if runtime is not None and len(xs) > 0 and runtime > xs[-1]:
        xs += [runtime]
        ys += [ys[-1]]
    ax.step(xs, ys, where='post', label=label)


def add_success_and_cost_plot(success_ax, cost_ax, metric_df, t_median_plot_start, label="", runtime=None, plot_range=False):
    plot_cost_plot_v2(cost_ax, df, metric_df, t_median_plot_start, time_step_sec=1, label=label, runtime=runtime, plot_range=plot_range)
    add_success_plot(success_ax, metric_df, label=label, runtime=runtime)

def draw_overlay_plot(input_path, output_path,map_path=None, draw_cars=False):
    targets = {}
    if os.path.isdir(input_path):
        for fn in os.listdir(input_path):
            # we exclude the final trajectory here, as it is a duplicate of the last intermediate trajectory
            if "trajectory" in fn and fn.endswith(".yaml") and not fn.startswith("final"):
                trajectory_id = int(fn.removesuffix(".yaml").split("_")[2])
                targets[trajectory_id] = os.path.join(input_path, fn)
    else:
        fn = input_path.split("/")[-1]
        trajectory_id = int(fn.removesuffix(".yaml").split("_")[2])
        targets[trajectory_id] = input_path

    blues = plt.get_cmap('Blues')

    if map_path is None:
        # if no map argument is given, check if the first trajectory file has a config param
        first_traj_dict = yaml.safe_load(open(targets[1], "r"))
        map_path = first_traj_dict["map_path"]

    animator = Animation(map_path)
    for traj_id in sorted(targets.keys()):
        if (traj_id - 1) % 1 != 0 and traj_id != len(targets.keys()):
            continue
        target = targets[traj_id]
        print(f"> {target}")

        shade_val = traj_id / len(targets.keys())
        color = blues(shade_val)

        traj_dict = yaml.safe_load(open(target, "r"))
        t_total = traj_dict["t_total_sec"]

        state_tuples = traj_dict["states"]
        if len(state_tuples) == 0:
            print(f"[red] target {target} does not contain any states!")
            continue
        states = [StateAckermannCarFirstOrder(x, y, theta) for (x, y, theta, *args) in state_tuples]
        animator.add_trajectory(states, color=color, label=f"iteration {traj_id}", draw_cars=draw_cars)

    if os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
        if os.path.isdir(input_path):
            animator.save_plot(os.path.join(output_path, f"trajectory_overlay.pdf"))
        else:
            animator.save_plot(os.path.join(output_path, f"trajectory_overlay_traj_{[key for key in targets.keys()][0]}{'_cars' if draw_cars else ''}.pdf"))
            # animator.show()
    else:
        animator.save_plot(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["trajectory_plots", "overlay_plot",
                                        "cost_evolution_plot",
                                        "cost_evolution_plot_comparison",
                                        "overlay_plots_all",
                                        "find_best"
                                        ])
    parser.add_argument("--input", "-i", required=False, type=str)
    parser.add_argument("--output", "-o", required=False, type=str)
    parser.add_argument("--map", "-m", required=False, type=str)
    parser.add_argument("--rebuild_metrics", required=False, action="store_true")
    parser.add_argument("--reparse", required=False, action="store_true")
    parser.add_argument("--config", required=False, type=str, help="path to the experiment config. "
                                                                   "will be used to load params, "
                                                                   "but the cli-args have precedent.")
    parser.add_argument("--median_50", action="store_true", help="if set, the median will be plotted for success > 50."
                                                                 "otherwise the plot starts at 100% success.")
    parser.add_argument("--plot_range", action="store_true", help="if set, the whole range (min - max cost) will be drawn")
    parser.add_argument("--runtime", required=False, type=float, help="max. runtime of the planner.")
    parser.add_argument("--draw_cars", action="store_true", help="if set, overlay plot will draw all car configs on the trajectory")
    args = parser.parse_args()

    if args.config is not None:
        config = yaml.safe_load(open(args.config, "r"))
        if args.runtime is None and "timeout_secs" in config:
            args.runtime = config["timeout_secs"]

    if args.cmd == "trajectory_plots":
        assert args.input is not None

        if args.output is not None:
            # input and output path need to both be a file or a dir
            assert os.path.isdir(args.output) == os.path.isdir(args.input)
            output_path = args.output
        else:
            if not os.path.isdir(args.input):
                output_path = args.input.split(".")[-2] + ".pdf"
            else:
                output_path = args.output

        if os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)

        targets = []
        if os.path.isdir(args.input):
            for fn in os.listdir(args.input):
                if "trajectory" in fn and fn.endswith(".yaml"):
                    targets.append(os.path.join(args.input, fn))
        else:
            fn = args.input.split("/")[-1]
            assert "trajectory" in fn and fn.endswith(".yaml")
            targets = [args.input]

        targets.sort()

        for target in targets:
            print(f"> {target}")
            traj_dict = yaml.safe_load(open(target, "r"))

            state_tuples = traj_dict["states"]
            if len(state_tuples) == 0:
                print(f"[red] target {target} contains no states")
                continue
            states = [StateAckermannCarFirstOrder(x, y, theta) for (x, y, theta, *args) in state_tuples]

            if args.map is None:
                map_path = traj_dict["map_path"]
            else:
                map_path = args.map

            animator = Animation(map_path)
            animator.add_trajectory(states)
            target_name = target.split("/")[-1].removesuffix(".yaml")
            if os.path.isdir(output_path):
                animator.save_plot(os.path.join(output_path, f"{target_name}.pdf"))
            else:
                animator.save_plot(output_path)

    elif args.cmd == "overlay_plot":
        if args.output is not None:
            output_path = args.output
        else:
            if os.path.isdir(args.input):
                output_path = args.input
            else:
                output_path = os.path.dirname(args.input)
        print(f"args.draw_cars: {args.draw_cars}")
        draw_overlay_plot(args.input, output_path, args.map, args.draw_cars)

    elif args.cmd == "overlay_plots_all":
        assert os.path.isdir(args.input)
        for subdir in track(os.listdir(args.input)):
            input_path = os.path.join(args.input, subdir)
            if not os.path.isdir(input_path):
                continue
            output_path = input_path
            draw_overlay_plot(input_path, output_path, args.map)

    # elif args.cmd == "cost_evolution_plot":
    #     print("[blue] < COST EVOLUTION PLOT >")
    #     assert os.path.isdir(args.input)
    #     data_frame_path = os.path.join(args.input, "results_dataframe.pkl")
    #     print(f"> df-path: <{data_frame_path}>")
    #     if os.path.exists(data_frame_path) and not args.reparse:
    #         df = pd.read_pickle(data_frame_path)
    #         print(f">> read existing df")
    #     else:
    #         df = construct_dataframe_for_experiment_run_dir(args.input)
    #         df.to_pickle(data_frame_path)
    #         print(f">> parsed and saved new df")
    #
    #
    #     metrics_df_path = os.path.join(args.input, "metrics_dataframe.pkl")
    #     print(f"> metrics path: <{metrics_df_path}>")
    #     if os.path.exists(metrics_df_path) and not args.rebuild_metrics:
    #         metrics_df = pd.read_pickle(metrics_df_path)
    #         print(">> read metrics df")
    #     else:
    #         metrics_df = construct_metrics_dataframe(df, time_step_sec=1)
    #         metrics_df.to_pickle(metrics_df_path)
    #         print(">> constructed and saved new metrics df")
    #
    #     fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
    #
    #     t_median_plot_start = metrics_df[metrics_df['success_perc'] >= 100]['t'].sort_values()[0]
    #     add_success_and_cost_plot(axs[1], axs[0], metrics_df, t_median_plot_start)
    #
    #     axs[0].legend()
    #     axs[0].set_ylabel("$\Delta t_{path} [s]$")
    #     axs[1].set_xlabel("$\Delta t[s]$")
    #     axs[1].set_ylabel("success [%]")
    #
    #     timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     output_path = args.output if args.output is not None \
    #         else os.path.join(args.input, f"cost_evolution_plot_{timestamp}.pdf")
    #
    #     plt.savefig(output_path)

    elif args.cmd == "cost_evolution_plot_comparison" or args.cmd == "cost_evolution_plot":
        assert os.path.isdir(args.input)
        if args.cmd == "cost_evolution_plot_comparison":
            experiment_dirs = [os.path.join(args.input, dir_name) for dir_name in os.listdir(args.input) if
                               os.path.isdir(os.path.join(args.input, dir_name))]
        else:
            experiment_dirs = [args.input]

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 5))

        for experiment_dir in track(sorted(experiment_dirs), description="Analysing and plotting experiments..."):
            experiment_name = "-".join(experiment_dir.split("/")[-1].split("_")[2:])
            data_frame_path = os.path.join(experiment_dir, "results_dataframe.pkl")
            print(f"> df-path: <{data_frame_path}>")
            if os.path.exists(data_frame_path) and not args.reparse:
                df = pd.read_pickle(data_frame_path)
                print(f">> read existing df")
            else:
                df = construct_dataframe_for_experiment_run_dir(experiment_dir)
                df.to_pickle(data_frame_path)
                print(f">> parsed and saved new df")

            metrics_df_path = os.path.join(experiment_dir, "metrics_dataframe.pkl")
            print(f"> metrics path: <{metrics_df_path}>")
            metrics_df = None
            if os.path.exists(metrics_df_path) and not args.rebuild_metrics:
                metrics_df = pd.read_pickle(metrics_df_path)
                print(">> read metrics df")
            else:
                metrics_df = construct_metrics_dataframe(df, time_step_sec=1)
                metrics_df.to_pickle(metrics_df_path)
                print(">> constructed and saved new metrics df")
            columns = metrics_df.columns
            if args.median_50:
                success_perc = metrics_df[metrics_df['success_perc'] >= 50]
            else:
                success_perc = metrics_df[metrics_df['success_perc'] >= 100]

            if len(success_perc) > 0:
                t_median_plot_start = success_perc['t'].sort_values().values.tolist()[0]
            else:
                t_median_plot_start = args.runtime
            add_success_and_cost_plot(axs[1], axs[0], metrics_df, t_median_plot_start,
                                      label=experiment_name, runtime=args.runtime, plot_range=args.plot_range)

        # from: https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
        handles, labels = axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # axs[1].legend()
        # Put a legend below current axis
        axs[1].legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.3),
                      fancybox=True, shadow=True, ncol=1)

        # axs[0].legend()
        axs[0].set_ylabel("$\Delta t_{path}\ [s]$")
        axs[1].set_xlabel("$t\ [s]$")
        axs[1].set_ylabel("success [%]")



        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = args.output if args.output is not None \
            else os.path.join(args.input, f"cost_evolution_plot_comparison_{timestamp}.pdf")

        plt.savefig(output_path, bbox_inches="tight")
        print(f"[green] Saved new plot at <{output_path}>")

    elif args.cmd == "find_best":
        assert os.path.isdir(args.input)
        experiment_dirs = [os.path.join(args.input, dir_name) for dir_name in os.listdir(args.input) if
                           os.path.isdir(os.path.join(args.input, dir_name))]

        return_dict = {}

        for experiment_dir in track(sorted(experiment_dirs), description="Analysing..."):
            data_frame_path = os.path.join(experiment_dir, "results_dataframe.pkl")
            print(f"> df-path: <{data_frame_path}>")
            if os.path.exists(data_frame_path) and not args.reparse:
                df = pd.read_pickle(data_frame_path)
                print(f">> read existing df")
            else:
                print(f"[red] no df found for {experiment_dir}")
                continue

            lowest_cost_row = df.sort_values(by="cost").iloc[0]
            return_dict[experiment_dir] = (lowest_cost_row["run_id"],lowest_cost_row["cost"])
        print("[blue]" + "-"*20)
        print("RESULTS")
        print("[blue]" + "-"*20)
        print(return_dict)
        print("[blue]" + "-"*20)







