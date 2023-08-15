import project_utils.utils as utils
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Tuple
from driver.ackermann_driver import AckermannCarFirstOrderDriver
import logging
from rl.environments.car import CustomEnv
from dynamics.AckermannCarDynamicsFirstOrder import AckermannCarDynamicsFirstOrder
from spaces.State import StateAckermannCarFirstOrder as State
from spaces.Action import ActionAckermannCarFirstOrder as Action
from rl.deployments.models import ActionAckermannFirstOrder
import uvicorn


class Trajectory(BaseModel):
    states: list[Tuple[float, float, float]]
    actions: list[Tuple[float, float]]


algo = "ppo"
model_path = "driver/star_models/ackermann/vel_ctrl/ppo_1678273546.zip"
model_path_onnx = "driver/star_models/ackermann/vel_ctrl/ppo_1678273546.onnx"
config_path = "driver/star_models/ackermann/vel_ctrl/config.yaml"


env = CustomEnv(config_path=config_path, trace_mode=False)

if algo == "ppo":
    from stable_baselines3 import PPO

    model = PPO.load(model_path, env)
elif algo == "a2c":
    from stable_baselines3 import A2C

    model = A2C.load(model_path, env)
elif algo == "sac":
    from stable_baselines3 import SAC

    model = SAC.load(model_path, env)
else:
    raise ValueError

app = FastAPI()
dynamics = AckermannCarDynamicsFirstOrder(config_path)
driver = AckermannCarFirstOrderDriver(
    config_path=config_path,
    model_path=model_path,
    dynamics=dynamics,
    model_path_onnx=model_path_onnx
)


# slower alternative
# @app.get("/predict")
# def get_prediction(x: float, y: float, theta: float):
#     obs = env.set_state(x, y, theta)
#     action, _states = model.predict(obs, deterministic=True)
#     return {
#         "lin_vel":str(action[0]),
#         "phi":str(action[1])
#     }

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
    logging.error(f"{request}: {exc_str}")
    content = {'status_code': 10422, 'message': exc_str, 'data': None}
    return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


@app.get("/predict_")
def get_prediction(x: float, y: float, theta: float):
    obs = env.set_state(x, y, theta)
    action, _states = model.predict(obs, deterministic=True)
    return ActionAckermannFirstOrder(
        lin_vel=action[0],
        phi=action[1]
    )


@app.get("/rollout")
def get_rollout(xi: float, yi: float, ti: float, xg: float, yg: float, tg: float):
    # states = []
    # actions = []
    # t_start = time.perf_counter_ns()
    # obs = env.set_state(x,y,theta)
    # x,y,theta = env.get_state()
    # states.append((x,y,theta))
    # done = False
    # while not done:
    #     action, _states = model.predict(obs, deterministic=True)
    #     lin_vel = project_utils.map_to_diff_range(env.action_space_low, env.action_space_high, env.min_lin_vel, env.max_lin_vel, action[0])
    #     phi = project_utils.map_to_diff_range(env.action_space_low, env.action_space_high, env.phi_min, env.phi_max, action[1])
    #     actions.append((lin_vel, phi))
    #     obs,_,done,_ = env.step(action)
    #     states.append(env.get_state())
    # t_end = time.perf_counter_ns()
    # print(f"rollout took {(t_end-t_start)/1_000_000} ms")
    # return Trajectory(states=states, actions=actions)
    global driver
    s_init = State(xi, yi, ti)
    s_goal = State(xg, yg, tg)

    _actions, _states = driver.propagate_rl(s_init, s_goal)
    actions = [(action.lin_vel, action.phi) for action in _actions]
    states = [(c.x, c.y, c.theta) for c in _states]
    return Trajectory(states=states, actions=actions)


@app.get("/rollout_")
def get_rollout(x: float, y: float, theta: float):
    states = []
    actions = []
    obs = env.set_state(x, y, theta)
    x, y, theta = env.get_state()
    states.append((x, y, theta))
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        lin_vel = utils.map_to_diff_range(env.action_space_low, env.action_space_high,
                                          env.dynamics.lin_vel_min, env.dynamics.lin_vel_max,
                                          action[0])
        phi = utils.map_to_diff_range(env.action_space_low, env.action_space_high,
                                      env.dynamics.phi_min, env.dynamics.phi_max, action[1])
        actions.append((lin_vel, phi))
        obs, _, done, _ = env.step(list(action))
        states.append(env.get_state())
    return Trajectory(states=states, actions=actions)


@app.get("/is-active")
def get_health_signal():
    return 1

if __name__ == "__main__":
    config = uvicorn.Config("deploy_via_fast_api:app", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()
