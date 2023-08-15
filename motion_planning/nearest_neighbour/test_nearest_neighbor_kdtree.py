import copy

import numpy as np

from motion_planning.Trees.Node import Node
from motion_planning.nearest_neighbour.nearest_neighbor_kdtree import NNStructure
from spaces.Action import ActionAckermannCarFirstOrder
from spaces.State import StateAckermannCarFirstOrder


class TestNNStructureAckermannFirstOrder:


    def test_distance_equality(self):
        nodes = []
        for _ in range(1000):
            x, y = np.random.uniform(0, 10, 2)
            theta = np.random.uniform(0, np.pi * 2)
            lin_vel = np.random.uniform(-0.1, 0.5)
            phi = np.random.uniform(- np.pi / 4, np.pi / 4)
            state = StateAckermannCarFirstOrder(x, y, theta)
            action = ActionAckermannCarFirstOrder(lin_vel, phi)
            nodes.append(Node(None, [action], [state]))
        query_state = nodes[0].get_final_state()
        nodes = nodes[1:]
        dists = [n.get_final_state().distance_to(query_state) for n in nodes]
        i_min = np.argmin(dists)
        min_dist = dists[i_min]
        min_node = nodes[i_min]
        nn_struct = NNStructure(1000)
        nn_struct.nodes = nodes
        nn_struct.build_tree()
        assert nn_struct.nr_inserted == 999
        min_dist_nn, min_node_nn = nn_struct.get_nearest_node_tree(query_state)
        assert min_node_nn == min_node
        assert np.isclose(min_dist_nn, min_dist)









    def test_add_thousand_random_root_nodes(self):
        nodes = []
        for _ in range(1000):
            x, y = np.random.uniform(0, 10, 2)
            theta = np.random.uniform(0, np.pi * 2)
            lin_vel = np.random.uniform(-0.1, 0.5)
            phi = np.random.uniform(- np.pi / 4, np.pi / 4)
            state = StateAckermannCarFirstOrder(x, y, theta)
            action = ActionAckermannCarFirstOrder(lin_vel, phi)
            nodes.append(Node(None, [action], [state]))

        max_waiting_nodes = 100
        nn_struct = NNStructure(max_waiting_nodes)
        c = 0
        for node in nodes[:50]:
            nn_struct.add_node(node)
        assert nn_struct.nr_inserted == 0
        assert len(nn_struct.nodes) == 50
        for node in nodes[50:150]:
            nn_struct.add_node(node)
        assert nn_struct.nr_inserted == 100
        assert len(nn_struct.nodes) == 150
        for node in nodes[150:]:
            nn_struct.add_node(node)
        assert len(nn_struct.nodes) == 1000
        assert nn_struct.nr_inserted == 1000

    def test_build_tree(self):
        nn_struct = NNStructure(100)
        nodes = []
        for _ in range(10):
            x, y = np.random.uniform(0, 10, 2)
            theta = np.random.uniform(0, np.pi * 2)
            lin_vel = np.random.uniform(-0.1, 0.5)
            phi = np.random.uniform(- np.pi / 4, np.pi / 4)
            state = StateAckermannCarFirstOrder(x, y, theta)
            action = ActionAckermannCarFirstOrder(lin_vel, phi)
            node = Node(None, [action], [state])
            nn_struct.add_node(node)
            nodes.append(node)

        x, y = np.random.uniform(0, 10, 2)
        theta = np.random.uniform(0, np.pi * 2)
        lin_vel = np.random.uniform(-0.1, 0.5)
        phi = np.random.uniform(- np.pi / 4, np.pi / 4)
        state = StateAckermannCarFirstOrder(x, y, theta)
        action = ActionAckermannCarFirstOrder(lin_vel, phi)
        query_node = Node(None, [action], [state])

        dist_arr = [n.get_final_state().distance_to(query_node.get_final_state()) for n in nodes[1:]]
        min_index = np.argmin(dist_arr)
        min_dist = dist_arr[min_index]
        min_node = nodes[1:][min_index]

        assert nn_struct.nr_inserted == 0
        assert nn_struct.tree is None
        nn_struct.build_tree()
        assert nn_struct.nr_inserted == 10
        assert nn_struct.tree is not None
        assert nn_struct.tree.n == 10
        min_node_nn = nn_struct.get_nearest_node(query_node.get_final_state())
        min_dist_nn = min_node_nn.get_final_state().distance_to(query_node.get_final_state())
        assert min_dist_nn == min_dist
        assert min_node_nn == min_node

    def test_add_node(self):
        nn_struct = NNStructure(100)
        x, y = np.random.uniform(0, 10, 2)
        theta = np.random.uniform(0, np.pi * 2)
        lin_vel = np.random.uniform(-0.1, 0.5)
        phi = np.random.uniform(- np.pi / 4, np.pi / 4)
        state = StateAckermannCarFirstOrder(x, y, theta)
        action = ActionAckermannCarFirstOrder(lin_vel, phi)
        node = Node(None, [action], [state])
        assert len(nn_struct.nodes) == 0
        nn_struct.add_node(node)
        assert len(nn_struct.nodes) == 1
        assert nn_struct.nodes[0] == node

    def test_max_extended_node(self):
        nn_struct = NNStructure(100)
        x, y = np.random.uniform(0, 10, 2)
        theta = np.random.uniform(0, np.pi * 2)
        lin_vel = np.random.uniform(-0.1, 0.5)
        phi = np.random.uniform(- np.pi / 4, np.pi / 4)
        state = StateAckermannCarFirstOrder(x, y, theta)
        action = ActionAckermannCarFirstOrder(lin_vel, phi)

        near_state = copy.copy(state)
        near_state.x = state.x + 0.1
        overextended_node = Node(None, [action], [near_state])
        overextended_node.extends = 50
        overextended_node.failed_extends = 50
        far_away_state = copy.copy(state)
        far_away_state.x = state.x + 5
        far_away_node = Node(None, [action], [far_away_state])
        nn_struct.add_node(overextended_node)
        nn_struct.add_node(far_away_node)
        nn_struct.build_tree()

        closest = nn_struct.get_nearest_node(state)
        assert closest == far_away_node


    def test_get_nearest_waiting_node():
        assert False
    def test_get_nearest_node_tree():
        assert False

    def test_get_nearest_node():
        assert False
