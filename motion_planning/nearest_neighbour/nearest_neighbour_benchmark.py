import importlib
import os
import time
import math
import random
from scenario_config_ackermann import Node, Config
import argparse
import nearest_neighbor as nn
import nearest_neighbor_kdtree as nnkd


def save_to_csv(data, path):
    with open(path, "w") as f:
        f.write("i,dt_query_ns\n")
        for (i, dt_query_ns) in data:
            f.write(f"{i},{dt_query_ns}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", help="number of points to add", default=5000, type=int)
    parser.add_argument("--n_update", help="number of noder per update", default=100, type=int)
    parser.add_argument("--kd_only", action="store_true")
    parser.add_argument("-l", help="length of the x axis", default=2)
    parser.add_argument("-w", help="width (length of the y axis)", default=2)
    args = parser.parse_args()


    max_abs_l = args.l/2
    max_abs_w = args.w/2

    def sample_config(w, l):
        x = (random.random()-0.5)*w
        y = (random.random()-0.5)*l
        theta = random.random()*math.pi*2
        return (x,y,theta)


    nn_struct = nn.NNStructure()
    nnkd_struct = nnkd.NNStructure(args.n_update)
    nn_times = []
    nnkd_times = []
    nodes = []

    for i in range(args.n):
        (x,y,theta) = sample_config(args.w, args.l)
        new_node = Node(Config(x,y,theta), [], None, [])
        nodes.append(new_node)
        nn_struct.addConfiguration(new_node)
        nnkd_struct.addConfiguration(new_node)
        [query_node] = random.sample(nodes, 1)

        start = time.perf_counter_ns()
        nearest_node_kd = nnkd_struct.nearest(query_node.config, dist_squared=False)
        stop = time.perf_counter_ns()
        nnkd_times.append((i, (stop-start)))

        if not args.kd_only:
            start = time.perf_counter_ns()
            nearest_node = nn_struct.nearest(query_node.config, dist_squared=False)
            stop = time.perf_counter_ns()
            nn_times.append((i, (stop-start)))
            if nearest_node != nearest_node_kd:
                print(f"query results do not match. kd: [{nearest_node_kd.config}] base: [{nearest_node.config}]")


        if i % 500 == 0 or i == (args.n-1):
            print(f"[{i}]: kdtree: {nnkd_times[-1][1]/1_000_000}ms" + ("" if args.kd_only else f" base: {nn_times[-1][1]/1_000_000}ms"))

    experiment_dir = "../kdrrt/nn_experiments"
    os.makedirs(experiment_dir, exist_ok=True)

    nnkd_path = os.path.join(experiment_dir, f"nnkd_{args.n_update}_{time.time()}.csv")
    save_to_csv(nnkd_times, nnkd_path)

    if not args.kd_only:
        nn_path = os.path.join(experiment_dir, f"nn_{time.time()}.csv")
        save_to_csv(nn_times, nn_path)

        
        



