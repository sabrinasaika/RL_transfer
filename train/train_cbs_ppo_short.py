import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cbs_env
from policies.masked_policy import MaskedMultiInputPolicy

def main():
    print("Creating CBS environment...")
    env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    eval_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)

    print("Creating PPO model...")
    # Use masked multi-input policy for CBS (obs+mask)
    model = PPO(MaskedMultiInputPolicy, env, verbose=1, n_steps=512, batch_size=32, learning_rate=3e-4)
    
    print("Starting training (short session)...")
    eval_cb = EvalCallback(eval_env, best_model_save_path="artifacts/policies/cbs_ppo/",
                           log_path="artifacts/policies/cbs_ppo/", eval_freq=1000,
                           deterministic=True, render=False)
    
    # Short training for testing (override with CBS_TRAIN_STEPS)
    total = int(os.environ.get("CBS_TRAIN_STEPS", "10000"))
    model.learn(total_timesteps=total, callback=eval_cb)
    model.save("artifacts/policies/cbs_ppo_final")
    print("Training completed! Model saved.")

if __name__ == "__main__":
    main()
