import pytest
import math
from spaces.Action import ActionAckermannCarFirstOrder
from spaces.State import CostStateAckermannCarFirstOrder
from motion_planning.kdrrt.kdrrt import generate_intermediate_nodes
from motion_planning.Trees.Node import Node
import numpy as np

@pytest.mark.parametrize("n_states, states_per_node", [(100,10),(200, 10), (555, 7)])
def test_intermediate_node_generation(n_states, states_per_node):
    actions_arr = np.random.uniform(-0.1, 0.1, (n_states-1, 2))
    pos = np.random.uniform(0, 5, (n_states, 2))
    theta = np.random.uniform(0, 2 * np.pi, n_states)
    first_pos = pos[0]
    cost = 0
    root_node = Node(None, [],
                     [CostStateAckermannCarFirstOrder(first_pos[0], first_pos[1], theta[0], cost_to_come=cost)])
    states = [CostStateAckermannCarFirstOrder(pos[i][0], pos[i][1], theta[i], i + 1) for i in range(len(pos))]
    actions = [ActionAckermannCarFirstOrder(lin_vel=lin_vel, phi=phi) for (lin_vel, phi) in actions_arr]

    nodes = generate_intermediate_nodes(root_node, actions, states[1:], states_per_node)
    assert nodes[-1].get_final_state() == states[-1]
    assert len(nodes) == math.ceil(n_states / states_per_node)
