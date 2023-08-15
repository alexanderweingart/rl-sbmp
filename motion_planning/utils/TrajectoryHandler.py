import datetime
from typing import List, Optional
from spaces.State import State
from spaces.Action import Action
from dynamics.Dynamics import Dynamics
import yaml
class TrajectoryHandler:
    @classmethod
    def store_trajectory(cls, dynamics: Dynamics, actions: List[Action], states: List[State],
                         s_target: State, output_path: str, map_path: Optional[str] = None,
                         t_start: Optional[datetime.datetime] = None, t_finished: Optional[datetime.datetime] = None):
        time_to_target = len(actions) * dynamics.delta_t
        final_distance_components = states[-1].distance_to_split_components(s_target)
        action_tuples = [a.vector().tolist() for a in actions]
        state_tuples = [s.vector().tolist() for s in states]
        return_dict = {
            "t_total_sec": time_to_target,
            "final_dist": [float(comp) for comp in final_distance_components],
            "actions": action_tuples,
            "states": state_tuples
        }
        if map_path is not None:
            return_dict["map_path"] = map_path

        if t_start is not None:
            return_dict["t_start"] = str(t_start)
        if t_finished is not None:
            return_dict["t_finished"] = str(t_finished)

        yaml.safe_dump(return_dict, open(output_path, "w"))
        return return_dict




