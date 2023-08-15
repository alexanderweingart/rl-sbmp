import torch as th
import yaml
from stable_baselines3 import PPO
from argparse import ArgumentParser

"""
Helper script to export stable baseline models to onnx files
based on: https://stable-baselines3.readthedocs.io/en/master/guide/export.html#export-to-onnx
"""


class OnnxablePolicy(th.nn.Module):
    def __init__(self, extractor, action_net, value_net):
        super().__init__()
        self.extractor = extractor
        self.action_net = action_net
        self.value_net = value_net

    def forward(self, observation):
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this. See `common.preprocessing.preprocess_obs`
        action_hidden, value_hidden = self.extractor(observation)
        actions = self.action_net(action_hidden)
        value = self.value_net(value_hidden)
        return actions, value


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("ppo_model_path", type=str, help="path to the sb3 model")
    parser.add_argument("config_path", type=str,
                        help="path to the environment/dynamics configuration file")
    parser.add_argument("output_path", type=str, help="path where the onnx file should be stored")
    args = parser.parse_args()

    robot_config = yaml.safe_load(open(args.config_path, "r"))
    dynamics = robot_config["dynamics"]

    if dynamics == "ackermann_car_first_order":
        from rl.environments.car import CustomEnv
    elif dynamics == "ackermann_car_second_order":
        from rl.environments.accl_car import CustomEnv
    else:
        raise NotImplementedError

    env = CustomEnv(args.config_path)
    print("-----------------------")
    print(f"dynamics: {dynamics}")
    print(f"env: {env}")
    print(f"obs-space: {env.observation_space}")
    print(f"action-space: {env.action_space}")
    print("-----------------------")

    model = PPO.load(args.ppo_model_path, env=env, device="cpu")

    onnxable_model = OnnxablePolicy(
        model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net
    )

    observation_size = model.observation_space.shape
    dummy_input = th.randn(1, *observation_size)
    th.onnx.export(
        onnxable_model,
        dummy_input,
        args.output_path,
        opset_version=9,
        input_names=["input"],
    )
