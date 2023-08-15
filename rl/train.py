import time
from rich import print
from utils.repo_checker import RepoChecker
import logging
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, filename=f"training-log-{int(time.time())}.log", filemode="a", format=FORMAT)
from rich.logging import RichHandler
logger = logging.getLogger("root")
logger.addHandler(RichHandler(show_level=True))
import yaml
import importlib
import os
import argparse
import logging.config

from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import HParam, Logger, configure


def train_and_store_model(model, seed, steps, eval_callback, model_path, algo, opath=None):
    tb_log_name = f"{algo}_{env.version_id}_s{seed}_lr{model.learning_rate}_ec{model.ent_coef}_g{model.gamma}"

    model.learn(total_timesteps=steps, progress_bar=True,
                reset_num_timesteps=False, tb_log_name=tb_log_name, callback=eval_callback)

    if opath is not None:
        logging.info(f"opath set --> saving to model to {opath}")
        model.save(opath)
    else:
        logging.info(f"saving model to {model_path}")
        model.save(model_path)

class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

    def _on_training_start(self) -> None:
        self.model: PPO|A2C
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "ent_coef": self.model.ent_coef,
            # "clip_range":self.model.clip_range,
            "batch_size":self.model.batch_size if type(self.model) == PPO else None,
            "max_grad_norm":self.model.max_grad_norm if type(self.model) != SAC else None,
            "vf_coef":self.model.vf_coef if type(self.model) != SAC else None,
            "use_sde":self.model.use_sde,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )
        self.logger.record(
            "curriculum_learning", bool(config['easy_training'])
        )
        self.logger.record(
            "max_n_starting_configs", 
            -1 if not config['easy_training'] else config['max_n_starting_configs'])

    def _on_step(self) -> bool:
        return True        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_module", type=str,
                        help="specifies the module where the environment (CustomEnv) is defined")
    parser.add_argument("--path", type=str, required=False)
    parser.add_argument("--opath", required=False, type=str)
    parser.add_argument("--best", action="store_true")
    parser.add_argument("--cohort", action="store_true")
    parser.add_argument("--algo", choices=["ppo", "a2c", "sac"], default="ppo")
    parser.add_argument("--learning_rate", type=float, required=False)
    parser.add_argument("--gamma", type=float, required=False)
    parser.add_argument("--buffer_size", type=float, required=False)
    parser.add_argument("--batch_size", type=float, required=False)
    parser.add_argument("--ent_coef", type=float, required=False)
    parser.add_argument("--use_sde", action="store_true")
    parser.add_argument("--sde_sample_freq", default=-1, type=int)
    parser.add_argument("--config", required=False, type=str)
    parser.add_argument("--seeds", required=False, help="list of pregenerated seeds")
    parser.add_argument("-n", type=int, default=500_000)
    parser.add_argument("-m", type=int, default=1)
    parser.add_argument("--eval_freq", default=10000, type=int, help="frequency of the evaluation")
    parser.add_argument("--eval_episodes", default=100, type=int, help="number of episodes per evaluation round")
    parser.add_argument("--log_file", required=False)
    parser.add_argument("--old_output_structure", action="store_true")
    parser.add_argument("--skip_git_check", action="store_true")
    parser.add_argument("--curiculum", action="store_true")

    args = parser.parse_args()
    if args.config is not None: # config params overwrite command line args
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)
            for key in config.keys():
                args.__dict__[key] = config[key]

    repo_checker = RepoChecker()
    if not args.skip_git_check and repo_checker.is_dirty(verbose=True):
        print("[red] Repo is dirty. Aborting.")
        exit()

    config_path_full = os.path.join(os.getcwd(), args.config)
    logging.info(f"loading config from [{config_path_full}]")


    env_mod = importlib.import_module(args.env_module)

    os.makedirs("logs", exist_ok=True)

#    logging.config.fileConfig("log.ini")

    if args.log_file is not None:
        logging.warning("the log_file config var is set. but this feature is currently not used!")
        # fh = logging.FileHandler(args.log_file, "a")
        # fh.setFormatter(formatter)
        # logger.addHandler(fh)


    logging.info("### trainings script started ###")
    logging.info("command line params:")
    args_dict = args.__dict__
    for key in args_dict.keys():
        logging.info(f"{key}: {args.__dict__[key]}")

    for j in range(max(args.m, 0 if args.seeds is None else len(args.seeds))):
        logging.info(f"++++  ROUND {j} +++++")
        time_stamp = time.time()
        time_stamp_str = str(time_stamp)

        if args.seeds is not None and j < len(args.seeds):
            seed = args.seeds[j]
        else:
            seed = int(time_stamp)

        logging.info(f"seed: {seed}")
        set_random_seed(seed)


        env = env_mod.CustomEnv(config_path = config_path_full)


        logging.info(f"environment id: {env.version_id}")
        if args.old_output_structure:
            base_dir = os.path.join(os.path.dirname(__file__), "data", f"{env.version_id}")
        else:
            commit_dir_name = "dirty_repo" if repo_checker.is_dirty() else repo_checker.get_commit_dir_name()
            base_dir = os.path.join(os.path.dirname(__file__), "data", commit_dir_name, f"{args.env_module.split('.')[-1]}_{args.config.split('/')[-2]}_{args.config.split('/')[-1]}")

        if j == 0:
            os.makedirs(base_dir, exist_ok=True)
            os.chdir(base_dir)

        env = Monitor(env, filename=f"{args.algo}_{seed}")

        if args.opath is not None:
            best_model_dir_path = os.path.dirname(args.opath)
            if not os.path.exists(best_model_dir_path):
                logging.warning(
                    f"The base dir of the output path does not exist. [{best_model_dir_path}]")
                logging.warning("exiting now.")
                exit(1)
            best_model_path = os.path.join(
                best_model_dir_path, f"{args.algo}_{seed}_best_model.zip")

        else:
            best_model_dir_path = f"models/{args.algo}_{seed}_best_model"
            best_model_path = os.path.join(
                best_model_dir_path, "best_model.zip")

        logging.info(f"path of the best model: {best_model_path}")

        if args.path is not None:
            logging.info("path was specified via parameter")
            model_path = args.path
        else:
            model_path = f"models/{args.algo}_{seed}.zip"

        logging.info(f"model_path: {model_path}")

        hp_callback = HParamCallback(config)
        callback = EvalCallback(env, log_path="log", best_model_save_path=best_model_dir_path,
                                warn=True, eval_freq=args.eval_freq, n_eval_episodes=args.eval_episodes,
                                callback_after_eval=hp_callback)


        if os.path.exists(model_path):
            if args.path is None:
                logging.info(
                    "model at this path already exists --> choosing this model to continue training")
            model = PPO.load(model_path, env)
        else:
            print("starting new model.")

            if args.algo == "ppo":
                model = PPO(
                    policy="MlpPolicy",
                    env=env,
                    tensorboard_log="log",
                    #     learning_rate=7.77e-05,
                    #     ent_coef=0.00429,
                    #     # ent_coef=0,
                    #     clip_range=0.1,
                    #     # batch_size=512,
                    #     batch_size=64,
                    #     gamma=0.999,
                    #     max_grad_norm=5,
                    #     vf_coef=0.19,
                        use_sde=args.use_sde,
                        sde_sample_freq=args.sde_sample_freq,
                    # policy_kwargs= dict(log_std_init=-3.29, ortho_init=False)
                    #    policy_kwargs= dict(ortho_init=False)
                )
            elif args.algo == "a2c":
                model = A2C(
                    policy="MlpPolicy",
                    env=env,
                    tensorboard_log="log",
                )
            elif args.algo == "sac":
                model = SAC(
                    "MlpPolicy",
                    env=env,
                    # batch_size=256,
                    # buffer_size=16384,
                    use_sde=True,
                    # n_timesteps= 1e6,
                    tensorboard_log="log",
                    learning_rate = 7.3e-4,
                    buffer_size = 65536,
                    batch_size = 1024,
                    ent_coef = 'auto',
                    gamma = 0.99,
                    tau = 0.02,
                    train_freq = 8,
                    gradient_steps = 8,
                    learning_starts = 10000,
                    # replay_buffer_kwargs: "dict(handle_timeout_termination=True)"
                    policy_kwargs = dict(log_std_init=-3, net_arch=[400, 300])
                )
            if args.algo != "sac":
                if args.learning_rate is not None:
                    model.learning_rate = args.learning_rate

                if args.gamma is not None:
                    model.gamma = args.gamma

                if args.ent_coef is not None:
                    model.ent_coef = args.ent_coef
                
                if args.batch_size is not None:
                    model.batch_size = args.batch_size
                
                if args.buffer_size is not None:
                    model.buffer_size = args.buffer_size

        model_dict = model.__dict__
        logging.info("model params:")
        for key in model_dict.keys():
            data = model_dict[key]
            logging.info(f">>> {key}: {data}")

        train_and_store_model(model, seed, args.n, callback,
                              model_path, args.algo, opath=args.opath)
