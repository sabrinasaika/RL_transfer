from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cbs_env

def main():
    env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    eval_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)

    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64, learning_rate=3e-4)
    eval_cb = EvalCallback(eval_env, best_model_save_path="artifacts/policies/cbs_ppo/",
                           log_path="artifacts/policies/cbs_ppo/", eval_freq=5000,
                           deterministic=True, render=False)
    model.learn(total_timesteps=300_000, callback=eval_cb)
    model.save("artifacts/policies/cbs_ppo_final")

if __name__ == "__main__":
    main()
