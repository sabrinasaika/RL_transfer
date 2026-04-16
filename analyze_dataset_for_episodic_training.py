"""
Analyze dataset to determine if augmentation is needed for episodic training.
Checks dataset sizes, class distribution, and ability to form episodes.
"""

import os
import sys
import numpy as np
from pathlib import Path
from collections import Counter

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def analyze_dataset(data_path=None, num_samples=1000):
    """
    Analyze dataset to determine if augmentation is needed.
    
    Args:
        data_path: Path to saved dataset (.npz file)
        num_samples: If no saved data, collect this many samples
    """
    print("=" * 80)
    print("DATASET ANALYSIS FOR EPISODIC TRAINING")
    print("=" * 80)
    
    # Load or collect data
    if data_path and os.path.exists(data_path):
        print(f"\nLoading dataset from: {data_path}")
        data = np.load(data_path, allow_pickle=True)
        
        if 'source_obs' in data:
            source_obs = data['source_obs'].tolist()
            target_obs = data['target_obs'].tolist() if 'target_obs' in data else []
            val_obs = data['val_obs'].tolist() if 'val_obs' in data else []
        else:
            # Legacy format: cw = source, cbs = target
            source_obs = data['cw_obs'].tolist() if 'cw_obs' in data else []
            target_obs = data['cbs_obs'].tolist() if 'cbs_obs' in data else []
            val_obs = []
    else:
        print(f"\nCollecting {num_samples} samples per domain (source=CW, target=CBS)...")
        from train_dapn_encoder_episodic import collect_observations_episodic
        source_obs, target_obs, val_obs, _, _ = collect_observations_episodic(
            num_samples=num_samples,
            save_path=None,
            val_fraction=0.2,
            seed=42,
        )
    
    # Convert to numpy arrays for analysis
    source_obs = np.array(source_obs) if len(source_obs) > 0 else np.array([])
    target_obs = np.array(target_obs) if len(target_obs) > 0 else np.array([])
    val_obs = np.array(val_obs) if len(val_obs) > 0 else np.array([])
    
    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}")
    print(f"Source domain (Ds): {len(source_obs)} samples")
    print(f"Target domain (Dd): {len(target_obs)} samples")
    print(f"Validation domain: {len(val_obs)} samples")
    
    # DAPN episodic training requirements
    print(f"\n{'='*80}")
    print("EPISODIC TRAINING REQUIREMENTS (from DAPN paper)")
    print(f"{'='*80}")
    print("For 5-way 5-shot classification:")
    print("  - Nmeta = 5 classes (test phase)")
    print("  - Nsc = 20 classes (training from Ds)")
    print("  - Ndc = 5 classes (training from Dd)")
    print("  - k = 5 samples per class (support set)")
    print("  - Query samples: ~15 per class")
    
    # Calculate requirements
    print(f"\n{'='*80}")
    print("EPISODE FORMATION REQUIREMENTS")
    print(f"{'='*80}")
    
    # For Ds (source domain)
    print("\n📊 Source Domain (Ds) Requirements:")
    nsc = 20  # Number of classes needed
    k = 5     # Shots per class
    query_per_class = 15
    samples_per_episode_ds = nsc * (k + query_per_class)  # 20 * 20 = 400
    min_samples_ds = samples_per_episode_ds  # At least one episode
    print(f"  - Need: {nsc} classes")
    print(f"  - Support set: {nsc} × {k} = {nsc * k} samples")
    print(f"  - Query set: {nsc} × {query_per_class} = {nsc * query_per_class} samples")
    print(f"  - Total per episode: {samples_per_episode_ds} samples")
    print(f"  - Minimum samples needed: {min_samples_ds}")
    print(f"  - Your samples: {len(source_obs)}")
    
    if len(source_obs) >= min_samples_ds:
        num_episodes_ds = len(source_obs) // samples_per_episode_ds
        print(f"  ✅ Can form ~{num_episodes_ds} episodes")
    else:
        print(f"  ❌ Cannot form even 1 episode!")
        print(f"     Need {min_samples_ds - len(source_obs)} more samples")
    
    # For Dd (target domain)
    print("\n📊 Target Domain (Dd) Requirements:")
    ndc = 5   # Number of classes needed
    k = 5     # Shots per class
    query_per_class = 15
    samples_per_episode_dd = ndc * (k + query_per_class)  # 5 * 20 = 100
    min_samples_dd = samples_per_episode_dd  # At least one episode
    print(f"  - Need: {ndc} classes")
    print(f"  - Support set: {ndc} × {k} = {ndc * k} samples")
    print(f"  - Query set: {ndc} × {query_per_class} = {ndc * query_per_class} samples")
    print(f"  - Total per episode: {samples_per_episode_dd} samples")
    print(f"  - Minimum samples needed: {min_samples_dd}")
    print(f"  - Your samples: {len(target_obs)}")
    
    if len(target_obs) >= min_samples_dd:
        num_episodes_dd = len(target_obs) // samples_per_episode_dd
        print(f"  ✅ Can form ~{num_episodes_dd} episodes")
    else:
        print(f"  ❌ Cannot form even 1 episode!")
        print(f"     Need {min_samples_dd - len(target_obs)} more samples")
    
    # Check class distribution (if we can infer classes)
    print(f"\n{'='*80}")
    print("OBSERVATION DIMENSIONS")
    print(f"{'='*80}")
    print("\n⚠️  IMPORTANT: The 8D you see is the UNIFIED translation layer!")
    print("   Raw observations are much larger:")
    print("   - Cyberwheel: Variable length (701D-70001D depending on max_num_hosts)")
    print("   - CyberBattleSim: Complex Dict with matrices/arrays (hundreds-thousands of dims)")
    print("\n   Your collected observations are likely:")
    print("   - Using unified 8D representation (if using ObservationTranslator)")
    print("   - OR full raw observations (if using UnifiedFullObsPreprocessor → 512D)")
    
    # Analyze observation statistics
    if len(source_obs) > 0:
        print(f"\nSource domain observation stats:")
        print(f"  Shape: {source_obs.shape}")
        print(f"  Dimension: {source_obs.shape[-1] if source_obs.ndim > 1 else 1}")
        if source_obs.ndim > 1:
            print(f"  Mean (first 8 dims): {source_obs.mean(axis=0)[:8]}")
            print(f"  Std (first 8 dims): {source_obs.std(axis=0)[:8]}")
        else:
            print(f"  Mean: {source_obs.mean()}")
            print(f"  Std: {source_obs.std()}")
    
    if len(target_obs) > 0:
        print(f"\nTarget domain observation stats:")
        print(f"  Shape: {target_obs.shape}")
        print(f"  Dimension: {target_obs.shape[-1] if target_obs.ndim > 1 else 1}")
        if target_obs.ndim > 1:
            print(f"  Mean (first 8 dims): {target_obs.mean(axis=0)[:8]}")
            print(f"  Std (first 8 dims): {target_obs.std(axis=0)[:8]}")
        else:
            print(f"  Mean: {target_obs.mean()}")
            print(f"  Std: {target_obs.std()}")
    
    print(f"\n{'='*80}")
    print("CLASS DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    print("\n⚠️  NOTE: Your domain uses continuous state vectors, not discrete classes.")
    print("   For episodic training, you'll need to:")
    print("   1. Cluster observations into 'classes' (e.g., using K-means)")
    print("   2. Or define classes based on state similarity")
    print("   3. Or use action-based grouping")
    
    # Final recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")
    
    needs_augmentation_dd = len(target_obs) < min_samples_dd
    
    if needs_augmentation_dd:
        print("\n❌ YES, you NEED augmentation for Dd (target domain):")
        print(f"   - Current samples: {len(target_obs)}")
        print(f"   - Required: {min_samples_dd}")
        print(f"   - Shortage: {min_samples_dd - len(target_obs)} samples")
        print("\n   Augmentation will:")
        print("   - Create D̂d from Dd using horizontal flips + 5 random crops")
        print("   - Increase samples to form episodes")
    else:
        print("\n✅ NO, you DON'T need augmentation for Dd (target domain):")
        print(f"   - Current samples: {len(target_obs)}")
        print(f"   - Required: {min_samples_dd}")
        print(f"   - Can form: ~{len(target_obs) // samples_per_episode_dd} episodes")
        print("\n   However, augmentation can still help:")
        print("   - Improve generalization")
        print("   - Create more diverse episodes")
        print("   - Better match DAPN paper implementation")
    
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print("1. Define 'classes' for your observations:")
    print("   - Option A: Cluster observations (K-means with n_clusters=20 for Ds, 5 for Dd)")
    print("   - Option B: Group by similar states (e.g., similar discovered_hosts, compromised_hosts)")
    print("   - Option C: Group by action outcomes or reward ranges")
    print("\n2. If augmentation needed:")
    print("   - For state vectors (NOT images!), augmentation means:")
    print("     * Adding Gaussian noise: obs + N(0, σ²)")
    print("     * Feature shuffling/permutation")
    print("     * Scaling/transformation: obs * scale + offset")
    print("     * Feature masking: randomly zero out some features")
    print("   - NOT traditional image augmentation (flips/crops don't apply here)")
    print("\n3. Implement episodic sampler:")
    print("   - CategoriesSampler for Ds (Nsc=20 classes)")
    print("   - CategoriesSampler for D̂d (Ndc=5 classes)")
    print("   - Split into support/query sets")
    
    return {
        'source_samples': len(source_obs),
        'target_samples': len(target_obs),
        'needs_augmentation': needs_augmentation_dd,
        'can_form_episodes_ds': len(source_obs) >= min_samples_ds,
        'can_form_episodes_dd': len(target_obs) >= min_samples_dd,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze dataset for episodic training")
    parser.add_argument("--data-path", type=str, default=None,
                       help="Path to saved dataset (.npz file)")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Number of samples to collect if no saved data")
    
    args = parser.parse_args()
    
    results = analyze_dataset(args.data_path, args.num_samples)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Needs augmentation: {results['needs_augmentation']}")
    print(f"Can form Ds episodes: {results['can_form_episodes_ds']}")
    print(f"Can form Dd episodes: {results['can_form_episodes_dd']}")
