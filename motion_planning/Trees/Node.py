from __future__ import annotations
from spaces.Action import Action
from spaces.State import State
from dynamics.AckermannCarDynamicsFirstOrder import StateAckermannCarFirstOrder
import numpy as np


class Node:
    states: list[State]  # states on the trajectory from parent to child. current state is the last in the list
    actions: list[Action]  # control applied to go from parent to child
    parent: Node

    def __init__(self, parent: Node | None, actions: list[Action],
                 states: list[State] | list[StateAckermannCarFirstOrder]):
        assert len(states) > 0, "states needs to include at least one state"
        self.parent = parent
        self.states = states
        self.actions = actions
        self.failed_extends = 0
        self.max_failed_extends = 20
        self.extends = 0

    def get_extension_failure_ratio(self):
        if self.extends == 0:
            return 0
        if self.failed_extends > self.max_failed_extends:
            return np.inf
        return self.failed_extends / self.extends

    def get_final_state(self):
        return self.states[-1]
