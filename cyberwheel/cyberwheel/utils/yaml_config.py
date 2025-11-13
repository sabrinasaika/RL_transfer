import yaml

from importlib.resources import files

class YAMLConfig:
    """
    Base class for YAML Configuration.
    """

    def __init__(self, config_file: str):
        self.config_file = files("cyberwheel.data.configs.environment").joinpath(
            config_file
        )

    def parse_config(self) -> None:
        with open(self.config_file, "r") as r:
            training_config = yaml.safe_load(r)
        self.__dict__.update(training_config)
        # Provide defaults expected by downstream code
        if not hasattr(self, "deterministic"):
            self.deterministic = False
        if not hasattr(self, "debug_mode"):
            self.debug_mode = False
        # Common training defaults
        self.experiment_name = getattr(self, "experiment_name", "CWRun")
        self.seed = getattr(self, "seed", 1)
        self.device = getattr(self, "device", "cpu")
        self.async_env = getattr(self, "async_env", False)
        self.track = getattr(self, "track", False)
        self.num_envs = getattr(self, "num_envs", 1)
        self.num_steps = getattr(self, "num_steps", 64)
        # Number of episodes to run during periodic checkpoint evaluations
        self.eval_episodes = getattr(self, "eval_episodes", 10)
        self.num_saves = getattr(self, "num_saves", 1)
        self.total_timesteps = getattr(self, "total_timesteps", 640)
        self.learning_rate = getattr(self, "learning_rate", 1e-4)
        self.ent_coef = getattr(self, "ent_coef", 0.0)
        self.clip_coef = getattr(self, "clip_coef", 0.1)
        self.norm_adv = getattr(self, "norm_adv", False)
        self.update_epochs = getattr(self, "update_epochs", 1)
        self.num_minibatches = getattr(self, "num_minibatches", 1)
        # Trainer expects this
        self.environment = getattr(self, "environment", "CyberwheelRL")
        # CW reward args commonly required
        self.valid_targets = getattr(self, "valid_targets", "all")
        self.red_reward_function = getattr(self, "red_reward_function", "reward_decoy_hits")
        self.blue_reward_function = getattr(self, "blue_reward_function", "reward_red_delay")
        # Minimal RL hyperparams used by RLHandler
        self.actor_lr = getattr(self, "actor_lr", self.learning_rate)
        self.critic_lr = getattr(self, "critic_lr", self.learning_rate)
        self.anneal_lr = getattr(self, "anneal_lr", False)
        self.restart_Tmult = getattr(self, "restart_Tmult", 1)
        self.min_lr = getattr(self, "min_lr", 1e-6)
        self.max_grad_norm = getattr(self, "max_grad_norm", 0.5)
        self.vf_coef = getattr(self, "vf_coef", 0.5)
        self.gamma = getattr(self, "gamma", 0.99)
        self.gae_lambda = getattr(self, "gae_lambda", 0.95)
        self.clip_vloss = getattr(self, "clip_vloss", False)
        self.target_kl = getattr(self, "target_kl", None)