import time
import math

from numpy import ndarray
from scipy.spatial import KDTree
import numpy as np
import random
from typing import Tuple
from motion_planning.Trees.Node import Node
from spaces.State import State
from spaces.State import CostStateAckermannCarSecondOrder
from spaces.State import CostStateAckermannCarFirstOrder
from typing import Tuple, Optional



class NNStructure:
    def __init__(self, max_n_waiting):
        self.nodes: list[Node] = []  # list of all nodes
        self.nr_inserted = 0  # index of the last inserted node in the tree
        self.tree: KDTree | None = None  # is built when the after the first n_update nodes have been added
        self.max_n_waiting = max_n_waiting
        self.max_cost = 0

    def build_tree(self):
        vecs = [np.asarray(n.get_final_state().vector_weighted())
                for n in self.nodes]
        self.tree = KDTree(np.array(vecs))
        self.nr_inserted = len(self.nodes)

    def add_node(self, new_node: Node):
        self.nodes.append(new_node)
        final_state = new_node.get_final_state()
        type_final_state = type(final_state)
        if type_final_state == CostStateAckermannCarSecondOrder or type_final_state == CostStateAckermannCarFirstOrder:
            cost_to_come = final_state.cost_to_come
            if cost_to_come > self.max_cost:
                self.max_cost = cost_to_come


        nr_of_waiting_nodes = len(self.nodes) - self.nr_inserted
        if nr_of_waiting_nodes == self.max_n_waiting:
            self.build_tree()

    def get_nearest_waiting_node(self, s: State) -> Tuple[float, Optional[Node]]:
        if self.nr_inserted == len(self.nodes):
            return np.inf, None
        vecs = [np.asarray(n.get_final_state().vector_weighted())
                for n in self.nodes[self.nr_inserted:]]
        query_vec = np.asarray(s.vector_weighted())
        dists = [math.dist(vec, query_vec) for vec in vecs]
        min_index = np.argmin(dists)
        # we have to add the offset, because the index is only based on the waiting list
        node_index = min_index + self.nr_inserted
        return dists[min_index], self.nodes[node_index]

    def get_nearest_node_tree(self, s: State) -> Tuple[float, Optional[Node]]:
        if self.nr_inserted == 0:
            return np.inf, None
        query_vec = np.asarray(s.vector_weighted())
        if len(self.nodes) == 1:
            vec = np.asarray(self.nodes[0].get_final_state().vector_weighted())
            return math.dist(vec, query_vec), self.nodes[0]
        [dist], [index] = self.tree.query([query_vec], workers=4)  # return only the nearest, use all cpus
        return dist, self.nodes[index]

    def get_nearest_node(self, s: State):
        # check the waiting nodes
        min_dist_waiting, min_node_waiting = self.get_nearest_waiting_node(s)
        # check the tree (slower)
        min_dist_tree, min_node_tree = self.get_nearest_node_tree(s)
        if min_dist_waiting < min_dist_tree:
            return min_node_waiting
        return min_node_tree




















# class NNStructureAckermannCarFirstOrder:
#     def __init__(self, n_update, theta_scaling, lin_vel_scaling, phi_scaling):
#         super().__init__(n_update)
#         self.l_side = []
#         self.l_side_vectors = []
#         self.l = []
#         self.l_vectors = []
#         self.tree: None | KDTree = None
#         self.n_update = n_update
#         self.theta_scaling = theta_scaling
#         self.lin_vel_scaling = lin_vel_scaling
#         self.phi_scaling = phi_scaling
#
#     def split_theta(self, theta):
#         x = math.cos(theta) * self.theta_scaling
#         y = math.sin(theta) * self.theta_scaling
#         return (x, y)
#
#     def build_scaled_vector(self, x,y,theta,lin_vel,phi):
#         theta_x, theta_y = self.split_theta(theta)
#         return (x, y, theta_x * self.theta_scaling, theta_y * self.theta_scaling, self.lin_vel_scaling * lin_vel, self.phi_scaling * phi)
#
#     def addConfiguration(self, n: Node):
#         self.l_side.append(n)
#         self.l_side_vectors.append(self.build_scaled_vector(n.config.x, n.config.y, n.config.theta, n.config.lin_vel, n.config.phi))
#         if len(self.l_side) >= self.n_update:
#             self.l += self.l_side
#             self.l_vectors += self.l_side_vectors
#             self.l_side = []
#             self.l_side_vectors = []
#             self.tree = KDTree(np.array(
#                 self.l_vectors))
#
#     def nearest(self, q: Config, dist_squared=False) -> Node:
#         query_vector = self.build_scaled_vector(q.x, q.y, q.theta, q.lin_vel, q.phi)
#
#         if dist_squared:
#             raise NotImplementedError
#
#         if self.tree is None:
#             if len(self.l_side_vectors) == 0:
#                 raise ValueError
#             dist_sides = [math.dist(query_vector, vec) for vec in self.l_side_vectors]
#
#             min_dist_index = np.argmin(dist_sides)
#             min_dist_side = dist_sides[min_dist_index]
#             node_min_dist_side = self.l_side[min_dist_index]
#             return node_min_dist_side
#
#         ([d], [i]) = self.tree.query(
#             [query_vector])
#         try:
#             tree_min_node = self.l[i]
#         except Exception as e:
#             print(f"d: {d} i: {i}")
#             raise e
#
#         if len(self.l_side) == 0:
#             return tree_min_node
#
#         dist_sides = [math.dist(query_vector, vec) for vec in self.l_side_vectors]
#         min_dist_index = np.argmin(dist_sides)
#         min_dist_side = dist_sides[min_dist_index]
#         node_min_dist_side = self.l_side[min_dist_index]
#
#         if d < min_dist_side:
#             return tree_min_node
#         return node_min_dist_side
#
#     def size(self):
#         return len(self.l) + len(self.l_side)
#
#     def sample_node(self):
#         all_nodes = self.l + self.l_side
#         return random.sample(all_nodes, 1)
#
#
# def recunstruct_path(end: Node):
#     path = []
#     n = end
#     while n is not None:
#         path.append(n)
#         n = n.parent
#     path.reverse()
#     return path
#
#
# if __name__ == "__main__":
#     print("Testing the NN datastruct")
#     print("generating 1000 random configs")
#     workspace = Workspace(0, 10, 0, 10)
#     sample_configs = [sample_config_uniform(workspace) for i in range(10000)]
#     sample_nodes = [Node(conf, [], None, []) for conf in sample_configs]
#     nn_struct = NNStructure(n_update=10)
#
#     t_start = time.perf_counter()
#     for node in sample_nodes:
#         nn_struct.addConfiguration(node)
#     t_end = time.perf_counter()
#     print(f"adding the configs took {(t_end-t_start):.4f} seconds")
#
#     new_node = Node(sample_config_uniform(workspace), [], None, [])
#     t_start = time.perf_counter()
#     result = nn_struct.nearest(new_node.config)
#     t_end = time.perf_counter()
#     print(f"nearest node found in {(t_end - t_start):.8f} seconds")
#
#
#
#
#
#
#
