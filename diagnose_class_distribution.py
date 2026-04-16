#!/usr/bin/env python3
"""
Diagnose why CategoriesSampler is only finding 5 classes with enough samples.

The issue: Even with 10,000 samples, K-means clustering creates uneven class sizes.
Only classes with >= 20 samples (k + query = 5 + 15) can be used for episodes.
"""

import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from adapters.episodic_training import cluster_observations, compress_labels


def diagnose_class_distribution(obs_list, n_clusters=20, k=5, query=15):
    """
    Diagnose class distribution after clustering.
    
    Args:
        obs_list: List of observations
        n_clusters: Number of clusters to create
        k: Shots per class
        query: Query samples per class
    """
    print("=" * 80)
    print("CLASS DISTRIBUTION DIAGNOSTIC")
    print("=" * 80)
    print(f"Total samples: {len(obs_list)}")
    print(f"Requested clusters: {n_clusters}")
    print(f"Required samples per class: {k + query} = {k + query}")
    print("=" * 80)
    
    # Convert to numpy array
    obs_array = np.array([np.array(obs) if not isinstance(obs, np.ndarray) else obs for obs in obs_list])
    print(f"\nObservation shape: {obs_array.shape}")
    
    # Cluster
    print("\n📊 Clustering observations...")
    labels = cluster_observations(obs_array, n_clusters=n_clusters, random_state=42)
    
    # Compress labels
    labels_compressed, unique_values = compress_labels(labels)
    print(f"\nCompressed to {len(unique_values)} unique classes")
    
    # Count samples per class
    unique, counts = np.unique(labels_compressed, return_counts=True)
    n_per = k + query
    
    print(f"\n📊 Class Distribution:")
    print(f"  Total classes: {len(unique)}")
    print(f"  Min samples per class: {counts.min()}")
    print(f"  Max samples per class: {counts.max()}")
    print(f"  Mean samples per class: {counts.mean():.1f}")
    print(f"  Median samples per class: {np.median(counts):.1f}")
    
    # Count classes with enough samples
    classes_with_enough = (counts >= n_per).sum()
    classes_with_few = (counts < n_per).sum()
    
    print(f"\n⚠️  Classes with >= {n_per} samples: {classes_with_enough}")
    print(f"   Classes with < {n_per} samples: {classes_with_few}")
    
    if classes_with_enough < n_clusters:
        print(f"\n❌ PROBLEM: Only {classes_with_enough} classes have enough samples!")
        print(f"   CategoriesSampler will only use these {classes_with_enough} classes.")
        print(f"   Each episode will have {classes_with_enough * n_per} samples instead of {n_clusters * n_per}.")
        
        # Show distribution
        print(f"\n📊 Detailed Distribution:")
        sorted_counts = np.sort(counts)[::-1]  # Descending
        print(f"  Top 10 largest classes: {sorted_counts[:10]}")
        print(f"  Bottom 10 smallest classes: {sorted_counts[-10:]}")
        
        # Calculate how many samples are "wasted" in small classes
        wasted_samples = counts[counts < n_per].sum()
        print(f"\n  Samples in classes with < {n_per} samples: {wasted_samples} ({wasted_samples/len(obs_list)*100:.1f}%)")
        
        # Suggest solutions
        print(f"\n💡 SOLUTIONS:")
        print(f"  1. Use --n-sc {classes_with_enough} to match available classes")
        print(f"  2. Use --label-mode action (creates more balanced classes)")
        print(f"  3. Collect more data (need ~{n_clusters * n_per * 2} samples for balanced clusters)")
        print(f"  4. Use balanced clustering (modify clustering to ensure min samples per class)")
    else:
        print(f"\n✅ All {classes_with_enough} classes have enough samples!")
    
    return classes_with_enough, counts


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose class distribution after clustering")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to .npz file with observations")
    parser.add_argument("--n-sc", type=int, default=20,
                       help="Number of classes (n_sc)")
    parser.add_argument("--k", type=int, default=5,
                       help="Shots per class")
    parser.add_argument("--query", type=int, default=15,
                       help="Query samples per class")
    parser.add_argument("--domain", type=str, default="source", choices=["source", "target"],
                       help="Which domain to diagnose")
    
    args = parser.parse_args()
    
    # Load data
    data = np.load(args.data, allow_pickle=True)
    if args.domain == "source":
        obs_list = data['source_obs'].tolist()
        print("Diagnosing SOURCE domain (Cyberwheel)")
    else:
        obs_list = data['target_obs'].tolist()
        print("Diagnosing TARGET domain (CyberBattleSim)")
    
    diagnose_class_distribution(obs_list, n_clusters=args.n_sc, k=args.k, query=args.query)
