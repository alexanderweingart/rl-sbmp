#!/usr/bin/env python3
import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
import matplotlib as mpl
from matplotlib import animation
from spaces.State import StateAckermannCarFirstOrder
from spaces.State import StateAckermannCarSecondOrder
from motion_planning.Trees.Node import Node
from typing import Optional, List
from dynamics.AckermannCarDynamicsFirstOrder import AckermannCarDynamicsFirstOrder
from dynamics.AckermannCarDynamicsSecondOrder import AckermannCarDynamicsSecondOrder
from rich import print
import copy


def draw_box_patch(ax, center, size, angle=0, **kwargs):
    xy = np.asarray(center) - np.asarray(size) / 2
    rect = Rectangle(xy, size[0], size[1], **kwargs)
    t = matplotlib.transforms.Affine2D().rotate_around(
        center[0], center[1], angle)
    rect.set_transform(t + ax.transData)
    ax.add_patch(rect)
    return rect


def gen_arrow_head_marker(angle):
    """generate a marker to plot with matplotlib scatter, plot, ...

    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers

    rot=0: positive x direction
    Parameters
    ----------
    rot : float
        rotation in degree
        0 is positive x direction

    Returns
    -------
    arrow_head_marker : Path
        use this path for marker argument of plt.scatter
    scale : float
        multiply a argument of plt.scatter with this factor got get markers
        with the same size independent of their rotation.
        Paths are autoscaled to a box of size -1 <= x, y <= 1 by plt.scatter
    """
    arr = np.array([[.1, .3], [.1, -.3], [2, 0], [.1, .3]])  # arrow shape
    rot_mat = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    arr = np.matmul(arr, rot_mat)  # rotates the arrow

    # scale
    x0 = np.amin(arr[:, 0])
    x1 = np.amax(arr[:, 0])
    y0 = np.amin(arr[:, 1])
    y1 = np.amax(arr[:, 1])
    scale = np.amax(np.abs([x0, x1, y0, y1]))
    codes = [mpl.path.Path.MOVETO, mpl.path.Path.LINETO, mpl.path.Path.LINETO, mpl.path.Path.CLOSEPOLY]
    arrow_head_marker = mpl.path.Path(arr, codes)
    return arrow_head_marker, scale


class Animation:
    def __init__(self, filename_env, filename_result: Optional[str] = None,
                 q_start_overwrite: Optional[StateAckermannCarFirstOrder] = None,
                 q_goal_overwrite: Optional[StateAckermannCarFirstOrder] = None, car_length=0.25, car_width=0.125):

        with open(filename_env) as env_file:
            env = yaml.safe_load(env_file)

        self.fig = plt.figure(figsize=(20,20))
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.ax.set_xlim(0, env["map"]["dimensions"][0])
        self.ax.set_ylim(0, env["map"]["dimensions"][1])
        self.car_length = car_length
        self.car_width = car_width

        self.roof_length_factor = 0.4  # length of the roof as a portion of the car length
        self.roof_width_factor = 0.7  # width of the roof as a portion of the car width
        self.roof_offset_factor = 0.2  # offset from the center on the x-axis (as a portion of the car length)

        self.car_color_body_target = (0xf7, 0xb7, 0x31)  # NYC taxi yellow
        self.car_color_roof_target = (0x0, 0x0, 0x0)  # black

        self.car_color_body = (0xe4, 0xc8, 0x8e)
        self.car_color_roof = (0x63, 0x63, 0x63)

        for obstacle in env["map"]["obstacles"] if env["map"]["obstacles"] is not None else []:
            if obstacle is None:
                continue
            if obstacle["type"] == "box":
                draw_box_patch(
                    self.ax, obstacle["center"], obstacle["size"], facecolor='gray', edgecolor='black')
            else:
                print("ERROR: unknown obstacle type")

        self.size = np.array([0.25, 0.175])

        if q_start_overwrite is not None:
            start = [q_start_overwrite.x, q_start_overwrite.y, q_start_overwrite.theta]
        else:
            start = env["robot"]["start"]

        if q_goal_overwrite is not None:
            goal = [q_goal_overwrite.x, q_goal_overwrite.y, q_goal_overwrite.theta]
        else:
            goal = env["robot"]["goal"]
        self.start = start
        self.goal = goal

        # draw_box_patch(self.ax, start[0:2], self.size,
        #                int(start[2]), facecolor='red')
        #
        # draw_box_patch(self.ax, goal[0:2], self.size,
        #                int(goal[2]), facecolor='none', edgecolor='red')
        self.draw_car(*start, colors=(self.car_color_body_target, self.car_color_roof_target))
        self.draw_car(*goal, colors=(self.car_color_body_target, self.car_color_roof_target))

        if filename_result is not None:
            with open(filename_result) as result_file:
                self.result = yaml.safe_load(result_file)

            T = 0
            for robot in self.result["result"]:
                T = max(T, len(robot["states"]))
            print("T", T)

            self.robot_patches = []
            for robot in self.result["result"]:
                state = robot["states"][0]
                patch = draw_box_patch(
                    self.ax, state[0:2], self.size, state[2], facecolor='blue')
                self.robot_patches.append(patch)

            self.anim = animation.FuncAnimation(self.fig, self.animate_func,
                                                frames=T,
                                                interval=100,
                                                blit=True)

    def draw_car(self, x, y, theta, colors):
        color_body = np.array(colors[0]) / 255
        color_roof = np.array(colors[1]) / 255

        xy_body = np.asarray((x, y)) - np.asarray((0, self.car_width)) / 2
        xy_roof = xy_body + np.asarray((((1 - self.roof_length_factor) / 2 - self.roof_offset_factor) * self.car_length,
                                        (1 - self.roof_width_factor) * 0.5 * self.car_width))
        # xy_roof = np.asarray((x-roof_length_factor*self.car_length, y)) \
        #           - np.asarray((self.car_length * roof_length_factor, self.car_width*roof_width_factor)) / 2
        rect_body = Rectangle(xy_body, self.car_length, self.car_width, color=color_body)
        rect_roof = Rectangle(xy_roof, self.car_length * self.roof_length_factor,
                              self.car_width * self.roof_width_factor, color=color_roof)
        t = matplotlib.transforms.Affine2D().rotate_around(
            x, y, theta)
        rect_body.set_transform(t + self.ax.transData)
        rect_roof.set_transform(t + self.ax.transData)
        self.ax.add_patch(rect_body)
        self.ax.add_patch(rect_roof)
        return rect_body

    def add_tree(self, nodes: List[Node]):
        """"
    takes a list of nodes (representing a tree) and 
    adds their line representation to the plot
    """
        referenced_nodes = []

        for n in nodes:
            if n in referenced_nodes:
                continue
            parent = n.parent
            while parent not in referenced_nodes and parent is not None:
                referenced_nodes.append(parent)
                parent = parent.parent

        leaves = []

        for n in nodes:
            if n not in referenced_nodes:
                leaves.append(n)

        already_line_nodes = []
        lines = []
        for leave in leaves:
            temp = leave
            line = []
            while temp is not None and temp not in already_line_nodes:
                line.append(temp)
                temp = temp.parent
            lines.append(line)

        for line in lines:
            for node in line:
                xs = [c.x for c in node.states[1:]]
                ys = [c.y for c in node.states[1:]]
                self.ax.plot(xs, ys, color="b", zorder=1)
                # self.ax.scatter([node.config.x],[node.config.y],color="green",s=5)
            for node in line:
                state = node.get_final_state()
                marker, scale = gen_arrow_head_marker(state.theta)
                markersize = 8
                self.ax.scatter(state.x, state.y, marker=marker, s=(markersize * scale) ** 2, color="green",
                                zorder=2)

    def add_trajectory(self, line: List[StateAckermannCarFirstOrder], color="r", draw_cars=False, label=None):
        if type(line[0]) is List:
            if len(line[0]) == 3:
                line = [StateAckermannCarFirstOrder(r[0], r[1], r[2]) for r in line]
            elif len(line[0]) == 5:
                line = [StateAckermannCarSecondOrder(r[0], r[1], r[2], r[3], r[4]) for r in line]
        for i in range(len(line) - 1):
            p1_x = line[i].x
            p1_y = line[i].y

            p2_x = line[i + 1].x
            p2_y = line[i + 1].y
            if draw_cars:
                self.draw_car(line[i+1].x, line[i+1].y, line[i+1].theta,
                              colors=(self.car_color_body, self.car_color_roof))
                self.draw_car(line[i+1].x, line[i+1].y, line[i+1].theta,
                              colors=(self.car_color_body, self.car_color_roof))
            else:
                self.ax.plot([p1_x, p2_x], [p1_y, p2_y], color=color, zorder=3)

        first_state = line[0]
        last_state = line[-1]

        self.draw_car(first_state.x, first_state.y, first_state.theta,
                      colors=(self.car_color_body, self.car_color_roof))
        self.draw_car(last_state.x, last_state.y, last_state.theta,
                      colors=(self.car_color_body, self.car_color_roof))

    def save(self, file_name, speed):
        self.anim.save(
            file_name,
            "ffmpeg",
            fps=10 * speed,
            dpi=200),

    def save_plot(self, file_name: str, legend: bool = False):
        if legend:
            self.fig.legend()
        self.fig.savefig(file_name, dpi=plt.gcf().dpi)
        plt.show()

    def show(self, block=True):
        plt.show(block=block)

    def animate_func(self, i):
        print(i)
        for robot, patch in zip(self.result["result"], self.robot_patches):
            state = robot["states"][i]
            center = state[0:2]
            angle = state[2]
            xy = np.asarray(center) - np.asarray(self.size) / 2
            patch.set_xy(xy)
            t = matplotlib.transforms.Affine2D().rotate_around(
                center[0], center[1], angle)
            patch.set_transform(t + self.ax.transData)
        return self.robot_patches

    def execute_trajectory(self, traj_path: str, robot_config_path: str, system: str, normalized_actions=False, step_through = False):
        traj = yaml.safe_load(open(traj_path,"r"))
        actions = traj["actions"]
        states = traj["states"]
        (x_init,y_init, theta_init) = self.start
        if system == "first-order-car":
            s_init = StateAckermannCarFirstOrder(*self.start)
            dynamics = AckermannCarDynamicsFirstOrder(robot_config_path)
        elif system == "second-order-car":
            s_init = StateAckermannCarSecondOrder(x_init, y_init, theta_init, 0, 0)
            dynamics = AckermannCarDynamicsSecondOrder(robot_config_path)
        else:
            raise NotImplementedError
        x = s_init.x
        y = s_init.y
        theta = s_init.theta
        for i, action in enumerate(actions):
            if normalized_actions:
                action = dynamics.convert_normalized_action(action, -1, 1)
            (x, y, theta) = dynamics.step([x, y, theta], action)
            self.draw_car(x, y, theta, [self.car_color_body, self.car_color_roof])
            if step_through:
                print(f"x: {x} y: {y} theta: {theta}")
                print(f"[blue] traj_file: x: {states[i+1][0]} y: {states[i+1][1]} theta: {states[i+1][2]}")
                self.show(block=False)
                input("press Enter to see the next")
        self.show()


def visualize(filename_env, filename_result=None, filename_video=None, q_start_overwrite=None, q_end_overwrite=None):
    anim = Animation(filename_env, filename_result, q_start_overwrite, q_end_overwrite)
    if filename_video is not None:
        anim.save(filename_video, 1)
    else:
        anim.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", help="input file containing map")
    subparser = parser.add_subparsers(dest="cmd")

    add_car = subparser.add_parser("add_car")
    add_car.add_argument("x", type=float)
    add_car.add_argument("y", type=float)
    add_car.add_argument("theta", type=float)

    exec_traj = subparser.add_parser("exec_traj")
    exec_traj.add_argument("traj", type=str)
    exec_traj.add_argument("--robot_config", "-rc", required=False, type=str)
    exec_traj.add_argument("--system", required=False, type=str)
    exec_traj.add_argument("--normalized_actions", action="store_true")
    exec_traj.add_argument("--step_through", action="store_true")

    args = parser.parse_args()

    if args.cmd == "add_car":
        visualizer = Animation(args.env, q_start_overwrite=None, q_goal_overwrite=None)
        if args.cmd == "add_car":
            visualizer.draw_car(args.x, args.y, args.theta, colors=(visualizer.car_color_body, visualizer.car_color_roof))
        visualizer.show(block=True)
    elif args.cmd == "exec_traj":
        visualizer = Animation(args.env, q_start_overwrite=None, q_goal_overwrite=None)
        visualizer.execute_trajectory(args.traj, args.robot_config, args.system, normalized_actions=args.normalized_actions, step_through=args.step_through)



if __name__ == "__main__":
    main()
