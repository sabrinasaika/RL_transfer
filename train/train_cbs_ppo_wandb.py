import os
from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cbs_env
import wandb


# WandbCallback doesn't exist in wandb.integration, using custom callback
class CustomWandbCallback(BaseCallback):
    """Simple callback to log metrics to wandb"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if 'episode' in info:
                    wandb.log({
                        'train/episode_reward': info['episode']['r'],
                        'train/episode_length': info['episode']['l']
                    })
        return True


def main():
    print("Starting CBS training with wandb")

    # Initialize wandb (offline mode if not authenticated)
    wandb.init(
        project="rl-transfer-cbs",
        name="cbs-ppo-training",
        dir="artifacts/wandb",
        config={
            "algorithm": "PPO",
            "env": "cyberbattlesim",
            "total_timesteps": int(os.environ.get("CBS_TRAIN_STEPS", "100000")),
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "clip_range": 0.2,
        },
        sync_tensorboard=True,
        mode="offline",  # Use offline mode
    )

    # Allow overriding total steps via env var for quick iterations
    total_steps = int(os.environ.get("CBS_TRAIN_STEPS", "100000"))

    # Create environments
    env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    eval_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    print("CBS environment created")

    # Create model with tensorboard logging and MultiInputPolicy for dict observations
    model = PPO(
        "MultiInputPolicy",  # Use MultiInputPolicy for dict observation space
        env,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        vf_coef=0.5,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=f"runs/{wandb.run.name}",  # Log to wandb run directory
    )

    # Create callbacks
    wandb_callback = CustomWandbCallback(verbose=2)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"artifacts/policies/cbs_ppo_wandb/",
        log_path=f"artifacts/policies/cbs_ppo_wandb/",
        eval_freq=max(5000, total_steps // 20),  # Evaluate at least 20 times during training
        deterministic=True,
        render=False,
    )

    print(f"Training for {total_steps} steps...")
    model.learn(
        total_timesteps=total_steps,
        callback=[wandb_callback, eval_callback]
    )

    # Ensure artifacts directory exists
    os.makedirs("artifacts/policies", exist_ok=True)
    model.save(f"artifacts/policies/cbs_ppo_wandb_{wandb.run.id}")
    print(f"Model saved to artifacts/policies/cbs_ppo_wandb_{wandb.run.id}")

    wandb.finish()
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
