#!/usr/bin/env python3
"""
Enhanced plot learning curves from tensorboard logs for CW training.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse


def plot_enhanced_learning_curve(log_dir):
    """
    Plot enhanced learning curves from tensorboard logs.
    
    Args:
        log_dir: Path to tensorboard log directory
    """
    print(f"Loading data from: {log_dir}")
    
    # Load the event accumulator
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    # Get available scalar tags
    scalar_tags = ea.Tags()['scalars']
    print(f"Available scalar tags: {scalar_tags}")
    
    # Extract data
    data = {}
    
    # Common metrics we want to plot
    metrics_to_plot = {
        'rollout/ep_len_mean': 'Training Episode Length',
        'rollout/ep_rew_mean': 'Training Episode Reward',
        'eval/mean_ep_length': 'Evaluation Episode Length',
        'eval/mean_reward': 'Evaluation Episode Reward',
        'train/policy_gradient_loss': 'Policy Loss',
        'train/value_loss': 'Value Loss',
        'train/entropy_loss': 'Entropy Loss',
        'train/approx_kl': 'Approximate KL Divergence',
        'train/explained_variance': 'Explained Variance'
    }
    
    for tag in scalar_tags:
        if tag in metrics_to_plot:
            scalar_events = ea.Scalars(tag)
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            data[tag] = {'steps': steps, 'values': values}
            print(f"Loaded {len(values)} data points for {tag}")
    
    if not data:
        print("No relevant metrics found!")
        return
    
    # Create a comprehensive plot with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Define subplot layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Episode Rewards (Training vs Evaluation)
    ax1 = fig.add_subplot(gs[0, :2])
    if 'rollout/ep_rew_mean' in data:
        steps = data['rollout/ep_rew_mean']['steps']
        values = data['rollout/ep_rew_mean']['values']
        ax1.plot(steps, values, 'b-', linewidth=2, label='Training Reward', alpha=0.8)
    
    if 'eval/mean_reward' in data:
        steps = data['eval/mean_reward']['steps']
        values = data['eval/mean_reward']['values']
        ax1.plot(steps, values, 'r-', linewidth=2, marker='o', markersize=6, label='Evaluation Reward')
    
    ax1.set_xlabel('Timesteps', fontsize=12)
    ax1.set_ylabel('Episode Reward', fontsize=12)
    ax1.set_title('Episode Reward Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 2: Episode Lengths
    ax2 = fig.add_subplot(gs[0, 2])
    if 'rollout/ep_len_mean' in data:
        steps = data['rollout/ep_len_mean']['steps']
        values = data['rollout/ep_len_mean']['values']
        ax2.plot(steps, values, 'g-', linewidth=2, label='Training Length')
    
    if 'eval/mean_ep_length' in data:
        steps = data['eval/mean_ep_length']['steps']
        values = data['eval/mean_ep_length']['values']
        ax2.plot(steps, values, 'orange', linewidth=2, marker='o', markersize=6, label='Evaluation Length')
    
    ax2.set_xlabel('Timesteps', fontsize=12)
    ax2.set_ylabel('Episode Length', fontsize=12)
    ax2.set_title('Episode Length Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Policy Loss
    ax3 = fig.add_subplot(gs[1, 0])
    if 'train/policy_gradient_loss' in data:
        steps = data['train/policy_gradient_loss']['steps']
        values = data['train/policy_gradient_loss']['values']
        ax3.plot(steps, values, 'purple', linewidth=2)
    ax3.set_xlabel('Timesteps', fontsize=12)
    ax3.set_ylabel('Policy Loss', fontsize=12)
    ax3.set_title('Policy Loss', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Value Loss
    ax4 = fig.add_subplot(gs[1, 1])
    if 'train/value_loss' in data:
        steps = data['train/value_loss']['steps']
        values = data['train/value_loss']['values']
        ax4.plot(steps, values, 'brown', linewidth=2)
    ax4.set_xlabel('Timesteps', fontsize=12)
    ax4.set_ylabel('Value Loss', fontsize=12)
    ax4.set_title('Value Loss', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Entropy Loss
    ax5 = fig.add_subplot(gs[1, 2])
    if 'train/entropy_loss' in data:
        steps = data['train/entropy_loss']['steps']
        values = data['train/entropy_loss']['values']
        ax5.plot(steps, values, 'pink', linewidth=2)
    ax5.set_xlabel('Timesteps', fontsize=12)
    ax5.set_ylabel('Entropy Loss', fontsize=12)
    ax5.set_title('Entropy Loss', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: KL Divergence
    ax6 = fig.add_subplot(gs[2, 0])
    if 'train/approx_kl' in data:
        steps = data['train/approx_kl']['steps']
        values = data['train/approx_kl']['values']
        ax6.plot(steps, values, 'cyan', linewidth=2)
    ax6.set_xlabel('Timesteps', fontsize=12)
    ax6.set_ylabel('KL Divergence', fontsize=12)
    ax6.set_title('Approximate KL Divergence', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Explained Variance
    ax7 = fig.add_subplot(gs[2, 1])
    if 'train/explained_variance' in data:
        steps = data['train/explained_variance']['steps']
        values = data['train/explained_variance']['values']
        ax7.plot(steps, values, 'olive', linewidth=2)
    ax7.set_xlabel('Timesteps', fontsize=12)
    ax7.set_ylabel('Explained Variance', fontsize=12)
    ax7.set_title('Explained Variance', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Summary Statistics
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Add summary text
    summary_text = "Training Summary:\n\n"
    if 'rollout/ep_rew_mean' in data:
        values = data['rollout/ep_rew_mean']['values']
        summary_text += f"Final Training Reward: {values[-1]:.2f}\n"
        summary_text += f"Best Training Reward: {max(values):.2f}\n"
        summary_text += f"Mean Training Reward: {np.mean(values):.2f}\n\n"
    
    if 'eval/mean_reward' in data:
        values = data['eval/mean_reward']['values']
        summary_text += f"Final Eval Reward: {values[-1]:.2f}\n"
        summary_text += f"Best Eval Reward: {max(values):.2f}\n"
        summary_text += f"Mean Eval Reward: {np.mean(values):.2f}\n\n"
    
    if 'train/policy_gradient_loss' in data:
        values = data['train/policy_gradient_loss']['values']
        summary_text += f"Final Policy Loss: {values[-1]:.4f}\n"
        summary_text += f"Mean Policy Loss: {np.mean(values):.4f}\n\n"
    
    if 'train/value_loss' in data:
        values = data['train/value_loss']['values']
        summary_text += f"Final Value Loss: {values[-1]:.1f}\n"
        summary_text += f"Mean Value Loss: {np.mean(values):.1f}\n"
    
    ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    # Add main title
    fig.suptitle('CW RL Policy Training - Learning Curves', fontsize=16, fontweight='bold')
    
    # Save plot
    output_file = 'cw_enhanced_learning_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Enhanced learning curve saved to: {output_file}")
    
    # Print detailed summary statistics
    print("\n" + "="*60)
    print("DETAILED TRAINING SUMMARY")
    print("="*60)
    
    for tag, info in data.items():
        values = info['values']
        if values:
            print(f"\n{metrics_to_plot[tag]}:")
            print(f"  Final: {values[-1]:.4f}")
            print(f"  Best:  {max(values):.4f}")
            print(f"  Worst: {min(values):.4f}")
            print(f"  Mean:  {np.mean(values):.4f}")
            print(f"  Std:   {np.std(values):.4f}")
    
    # Try to show the plot if in interactive mode
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot enhanced learning curves from tensorboard logs')
    parser.add_argument('--log-dir', type=str, 
                       default='/home/ssaika/rl-transfer-sec-clean/runs/cw-ppo-training/PPO_2',
                       help='Path to tensorboard log directory')
    args = parser.parse_args()
    
    plot_enhanced_learning_curve(args.log_dir)
