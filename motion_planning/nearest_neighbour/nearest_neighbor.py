from scenario_config import Config
import math
from scenario_config import Point, distance, Workspace, Node, distance_squared
from scenario_config import sample_config_uniform
import numpy as np
import random

class NNStructure:
    l: list[Node]

    def __init__(self):
        self.l = []

    def addConfiguration(self, n: Node):
        self.l.append(n)
    
    def nearestK(self,q: Config, K: int) -> list[Node]:
        def extract_second_elem(x: tuple):
            return x[1]
        distances = []
        min_dist = np.inf
        min_dist_node = None
        for n in self.l:
            if n.config.x == q.x and n.config.y == q.y and n.config.theta == q.theta:
                continue
            d = distance(q,n.config)
            distances.append((n, d))
            if min_dist_node is None or d < min_dist:
                min_dist = d
                min_dist_node = n
            # print("appending tuple to distances")
        if K == 1:
            return [min_dist_node]
        distances.sort(key=extract_second_elem, reverse=False)
        if K > len(distances):
            K = len(distances)
        # print(f"K : {K}")
        first_k = distances[:K]
        ns = [n for (n,d) in first_k]
        # print(f"len of nodes in nearestK: {len(ns)}")
        return ns
    

    def nearest(self, q: Config, dist_squared = True) -> Node:
        if dist_squared:
            distances = [distance_squared(q, nd.config) for nd in self.l]
        else:
            distances = [math.dist((q.x,q.y),(nd.config.x, nd.config.y)) for nd in self.l]
        return self.l[np.argmin(distances)]



    def nearestR(self,q: Config, r: float):
        results = []
        for n in self.l:
            if n.point == q:
                continue
            if distance(n.config,q) <= r:
                results.append(n)
        return results
    
    def size(self):
        return len(self.l)
    
    def sample_node(self):
        return random.sample(self.l, 1)


def recunstruct_path(end: Node):
    path = []
    n = end
    while n is not None:
        path.append(n)
        n = n.parent
    path.reverse()
    return path


if __name__ == "__main__":
    print("Testing the NN datastruct")
    print("generating 10 random points.")
    workspace = Workspace(0,10,0,10)
    sample_points = [sample_config_uniform(workspace) for i in range(10)]
    idxs = [i for i in range(10)]
    random.shuffle(idxs)
    nodes = []
    nodes.append(Node(sample_points[idxs[0]], None))
    for idx in idxs[1:]:
        nodes.append(Node(sample_points[idx], nodes[-1]))
    print("Done")
    print("Reconstructing path:")
    path = recunstruct_path(nodes[-1])
    idx = 0
    for n in path:
        print(f"{idx}: ({n.point.x:.2f},{n.point.y:.2f})")
        idx += 1


    







