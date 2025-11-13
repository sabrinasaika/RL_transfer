import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import json


def plot_learning_curve(run_path=None):
    """
    Plot learning curves from a wandb run.
    
    Args:
        run_path: Path to wandb offline run directory or None to use latest
    """
    # If no run_path specified, use the latest run
    if run_path is None:
        # Find the latest offline run
        wandb_dir = "wandb"
        if os.path.exists(wandb_dir):
            runs = [d for d in os.listdir(wandb_dir) if d.startswith("offline-run-")]
            if runs:
                latest_run = max(runs, key=lambda x: os.path.getctime(os.path.join(wandb_dir, x)))
                run_path = os.path.join(wandb_dir, latest_run)
                print(f"Using latest run: {latest_run}")
        else:
            print("No wandb directory found!")
            return
    
    print(f"Loading data from: {run_path}")
    history_file = os.path.join(run_path, "files", "wandb-history.jsonl")
    if not os.path.exists(history_file):
        print(f"History file not found at: {history_file}")
        return
    
    # Load data
    data = []
    with open(history_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"Loaded {len(data)} data points")
    
    # Extract metrics
    steps = [d.get('_step', 0) for d in data if '_step' in d]
    episode_rewards = [d.get('rollout/ep_rew_mean', 0) for d in data if 'rollout/ep_rew_mean' in d]
    episode_lengths = [d.get('rollout/ep_len_mean', 0) for d in data if 'rollout/ep_len_mean' in d]
    eval_rewards = [d.get('eval/mean_reward', 0) for d in data if 'eval/mean_reward' in d]
    eval_steps = [d.get('_step', 0) for d in data if 'eval/mean_reward' in d]
    policy_losses = [d.get('train/policy_gradient_loss', 0) for d in data if 'train/policy_gradient_loss' in d]
    value_losses = [d.get('train/value_loss', 0) for d in data if 'train/value_loss' in d]
    loss_steps = [d.get('_step', 0) for d in data if 'train/policy_gradient_loss' in d]
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode reward
    if episode_rewards:
        axes[0, 0].plot(steps[:len(episode_rewards)], episode_rewards, linewidth=2)
        axes[0, 0].set_xlabel('Timesteps', fontsize=12)
        axes[0, 0].set_ylabel('Mean Episode Reward', fontsize=12)
        axes[0, 0].set_title('Episode Reward Over Time', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Evaluation reward
    if eval_rewards:
        axes[0, 1].plot(eval_steps, eval_rewards, marker='o', color='orange', markersize=6, linewidth=2)
        axes[0, 1].set_xlabel('Timesteps', fontsize=12)
        axes[0, 1].set_ylabel('Mean Evaluation Reward', fontsize=12)
        axes[0, 1].set_title('Evaluation Reward Over Time', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Episode length
    if episode_lengths:
        axes[1, 0].plot(steps[:len(episode_lengths)], episode_lengths, linewidth=2, color='green')
        axes[1, 0].set_xlabel('Timesteps', fontsize=12)
        axes[1, 0].set_ylabel('Mean Episode Length', fontsize=12)
        axes[1, 0].set_title('Episode Length Over Time', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Loss metrics
    if policy_losses and loss_steps:
        axes[1, 1].plot(loss_steps, policy_losses, label='Policy Loss', linewidth=2)
    if value_losses and loss_steps:
        axes[1, 1].plot(loss_steps, value_losses, label='Value Loss', linewidth=2)
    axes[1, 1].set_xlabel('Timesteps', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].set_title('Training Losses', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'learning_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Learning curve saved to: {output_file}")
    
    # Print summary statistics
    if episode_rewards:
        print(f"\nýmEpisode Reward Summary:")
        print(f"  Final: {episode_rewards[-1]:.2f}")
        print(f"  Best:  {max(episode_rewards):.2f}")
        print(f"  Mean:  {np.mean(episode_rewards):.2f}")
    
    if eval_rewards:
        print(f"\nEvaluation Reward Summary:")
        print(f"  Final: {eval_rewards[-1]:.2f}")
        print(f"  Best:  {max(eval_rewards):.2f}")
        print(f"  Mean:  {np.mean(eval_rewards):.2f}")
    
    # Try to show the plot if in interactive mode, otherwise just save it
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot learning curves from wandb')
    parser.add_argument('--run-path', type=str, default=None, help='Path to wandb run directory')
    args = parser.parse_args()
    
    plot_learning_curve(args.run_path)
