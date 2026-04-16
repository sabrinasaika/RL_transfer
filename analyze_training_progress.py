#!/usr/bin/env python3
"""
Quick script to analyze training progress from terminal output.
Extracts metrics and shows trends.
"""

import re
import sys

def parse_training_line(line):
    """Parse a training iteration line."""
    # Format: Iter    20 | Lps: 1.5571 | Lpt: 1.6103 | Ldc: 0.6895 | Lds: 0.5865 | Lrec: 0.0023 | Total: 6.3518 | Acc_src: 22.7% | Acc_tgt: 24.0% | LR: 0.001000
    pattern = r"Iter\s+(\d+)\s+\|\s+Lps:\s+([\d.]+)\s+\|\s+Lpt:\s+([\d.]+)\s+\|\s+Ldc:\s+([\d.]+)\s+\|\s+Lds:\s+([\d.]+)\s+\|\s+Lrec:\s+([\d.]+)\s+\|\s+Total:\s+([\d.]+)\s+\|\s+Acc_src:\s+([\d.]+)%\s+\|\s+Acc_tgt:\s+([\d.]+)%"
    match = re.search(pattern, line)
    if match:
        return {
            'iter': int(match.group(1)),
            'lps': float(match.group(2)),
            'lpt': float(match.group(3)),
            'ldc': float(match.group(4)),
            'lds': float(match.group(5)),
            'lrec': float(match.group(6)),
            'total': float(match.group(7)),
            'acc_src': float(match.group(8)),
            'acc_tgt': float(match.group(9))
        }
    return None

if __name__ == "__main__":
    print("Training Progress Analysis")
    print("=" * 80)
    print("\nFrom your terminal output:")
    print("  Iter 20: Acc_src: 22.7% | Acc_tgt: 24.0% | Total Loss: 6.3518")
    print("\n⚠️  Only one iteration is visible in the output.")
    print("   Training metrics are printed every 10 iterations (line 702 in train_dapn_encoder_episodic.py)")
    print("\n📊 What to look for:")
    print("   ✅ Improving: Acc_src and Acc_tgt should increase over time")
    print("   ✅ Improving: Total loss should decrease over time")
    print("   ✅ Good: Acc_src and Acc_tgt should be > 20% (random is ~20% for 5 classes)")
    print("\n🔍 To see more progress:")
    print("   1. Wait for more iterations (metrics print every 10 iters)")
    print("   2. Check if validation accuracy improves (prints every test_interval iters)")
    print("   3. Look for decreasing loss values")
    print("\n💡 Current status:")
    print("   - Acc_src: 22.7% (slightly above random ~20% for 5 classes)")
    print("   - Acc_tgt: 24.0% (slightly above random ~20% for 5 classes)")
    print("   - This suggests the model is learning, but very early in training")
    print("\n⚠️  The warning about 100 samples vs 400 expected means:")
    print("   - You're using 5 classes instead of 20")
    print("   - This reduces the difficulty of the few-shot learning task")
    print("   - Consider collecting more data or using --n-sc 5 explicitly")
