from spaces.State import State
class MetaMotionPlanner:
    """
    Meta Class for Motion Planners
    """
    def __init__(self, planner_id: str, planner_dir: str, config_path: str):
        """
        Default constructur. Only initialises meta data which is interesting for all planners
        @param planner_id: unique id string for this planner
        @param planner_dir: directory of the planner (used to perform the git check)
        @param config_path: path to the configuration, that is used. this will be archived in experiments.
                            (the configuration should encompass all parameters of the planner, to enable easy
                            reproducability of the experiments)
        """
        self.planner_id = planner_id
        self.planner_dir = planner_dir
        self.config_path = config_path

        self.s_init: State
