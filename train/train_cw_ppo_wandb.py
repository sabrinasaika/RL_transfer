import os
import sys
sys.path.append('/home/ssaika/rl-transfer-sec-clean')
sys.path.append('/home/ssaika/rl-transfer-sec-clean/cyberwheel')

from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback
from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cw_env
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
    print("Starting CW training with wandb")

    # Initialize wandb (offline mode if not authenticated)
    wandb.init(
        project="rl-transfer-cw",
        name="cw-ppo-training",
        dir="artifacts/wandb",
        config={
            "algorithm": "PPO",
            "env": "cyberwheel",
            "total_timesteps": int(os.environ.get("CW_TRAIN_STEPS", "50000")),
            "learning_rate": 3e-4,
            "n_steps": 512,
            "batch_size": 256,
            "gamma": 0.995,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "clip_range": 0.2,
        },
        sync_tensorboard=True,
        mode="offline",  # Use offline mode
    )

    # Allow overriding total steps via env var for quick iterations
    total_steps = int(os.environ.get("CW_TRAIN_STEPS", "50000"))

    # Create environments
    env = UnifiedSecEnv("cw", cw_factory=make_cw_env)
    eval_env = UnifiedSecEnv("cw", cw_factory=make_cw_env)
    print("CW environment created")

    # Create model with tensorboard logging
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.995,
        ent_coef=0.01,
        vf_coef=0.5,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=f"runs/{wandb.run.name}",  # Log to wandb run directory
    )

    # Create callbacks
    wandb_callback = CustomWandbCallback(verbose=2)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"artifacts/policies/cw_ppo_wandb/",
        log_path=f"artifacts/policies/cw_ppo_wandb/",
        eval_freq=max(1000, total_steps // 10),  # Evaluate at least 10 times during training
        deterministic=True,
        render=False,
    )

    # Checkpoint callback: save model snapshots at the same cadence as evals
    eval_freq = max(1000, total_steps // 10)
    ckpt_dir = os.path.join("artifacts", "policies", "cw_ppo_wandb", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=ckpt_dir,
        name_prefix="cw_ppo_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    print(f"Training for {total_steps} steps...")
    model.learn(
        total_timesteps=total_steps,
        callback=[wandb_callback, eval_callback, checkpoint_callback]
    )

    # Ensure artifacts directory exists
    os.makedirs("artifacts/policies", exist_ok=True)
    model.save(f"artifacts/policies/cw_ppo_wandb_{wandb.run.id}")
    print(f"Model saved to artifacts/policies/cw_ppo_wandb_{wandb.run.id}")

    wandb.finish()
    print("Training completed successfully!")


if __name__ == "__main__":
    main()


