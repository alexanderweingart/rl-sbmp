from sympy import *
from argparse import ArgumentParser
import math
def print_theory():
    theta_g, x_g, y_g, theta_b, x_b, y_b = symbols('theta_{g} x_{g} y_{g} theta_b x_b y_b')
    x_t,y_t,theta_t = symbols('x^G y^G theta^G')

    T_G_W = Matrix([[cos(theta_g), sin(theta_g), -cos(theta_g)*x_g - sin(theta_g)*y_g],
                [-sin(theta_g), cos(theta_g), sin(theta_g)*x_g - cos(theta_g)*y_g], [0, 0, 1]])
    T_W_B = Matrix([[cos(theta_b), -sin(theta_b), x_b],[sin(theta_b), cos(theta_b), y_b],[0,0,1]])
    T_W_G = Matrix([[cos(theta_g), -sin(theta_g), x_g],[sin(theta_g), cos(theta_g), y_g],[0,0,1]])
    T_B_W = Matrix([[cos(theta_b), sin(theta_b), -cos(theta_b)*x_b - sin(theta_b)*y_b],
                [-sin(theta_b), cos(theta_b), sin(theta_b)*x_b - cos(theta_b)*y_b], [0, 0, 1]])


    T_G_B = T_G_W.multiply(T_W_B)
    T_B_G = T_B_W.multiply(T_W_G)
    print("-"*20)
    print("T_G_B:")
    pprint(simplify(T_G_B))
    print("latex:")
    print(latex(simplify(T_G_B)))
    print("-"*20)
    print("T_B_G:")
    pprint(simplify(T_B_G))
    print("latex:")
    print(latex(simplify(T_B_G)))
    print("chosen target frame (relative to world frame):")
    pprint(T_W_G)
    print("current base frame (relative to world frame): ")
    pprint(T_W_B)
    print("base frame relative to target frame: ")
    T_G_B =  simplify(T_G_W * T_W_B)
    pprint(T_G_B)
    pose_in_goal_frame = Matrix([[cos(theta_t), -sin(theta_t), x_t],[sin(theta_t), cos(theta_t), y_t],[0,0,1]])
    print("pose in goal frame at time step t: ")
    pprint(pose_in_goal_frame)
    print("to transform it back to world frame it has to be multiplied with T^W_G: ")
    pprint(simplify(T_W_G * pose_in_goal_frame))
    print("latex:")
    print(latex(simplify(T_W_G * pose_in_goal_frame)))




def transform_to_target_frame(x,y,theta,x_t,y_t,theta_t):
    new_theta = (theta - theta_t) % (2*math.pi)
    new_x = math.cos(theta_t) * (x-x_t) + math.sin(theta_t) * (y-y_t)
    new_y = math.sin(theta_t) * (x_t-x) + math.cos(theta_t) * (y-y_t)
    return (new_x, new_y, new_theta)

def transform_from_target_frame(x,y,theta,x_t,y_t,theta_t):
    new_theta = (theta + theta_t) % (2*math.pi)
    new_x = x*math.cos(theta_t) + x_t - y*math.sin(theta_t)
    new_y = x*math.sin(theta_t) + y_t + y*math.cos(theta_t)
    return (new_x, new_y, new_theta)


def transformation_tests():
    target_pose = (0, 1, math.pi)
    test_pose_1 = (0, 0, 0)
    test_pose_2 = (1,1,0)
    test_poses = [test_pose_1, test_pose_2]
    (x_t, y_t, theta_t) = target_pose
    for pose in test_poses:
        print(f"pose: {pose}")
        (x,y,theta) = pose
        (_x,_y,_theta) = transform_to_target_frame(x,y,theta, x_t, y_t, theta_t)
        print(f"in target frame: {[_x,_y,_theta]}")
        (x,y,theta) = transform_from_target_frame(_x,_y,_theta, x_t, y_t, theta_t)
        print(f"transformed back:{[x,y,theta]}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("cmd", choices=["theory", "tests"])
    args = parser.parse_args()

    if args.cmd == "theory":
        print_theory()
    elif args.cmd == "tests":
        transformation_tests()

