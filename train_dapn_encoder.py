"""
Train DAPN encoder with:
  1) Full raw observation collection (CW + CBS)
  2) Paired-state collection (optional) WITHOUT stepping to preserve pairing
  3) 512D fixed-size preprocessing
  4) Dataset-based normalization (mean/std) instead of /100 clipping
  5) Balanced batches (always half source, half target)
  6) Proper DANN objective (no "1.0 - BCE"). Encoder is trained to confuse discriminator via -lambda * adv_loss.

Usage examples:
  python train_dapn_encoder_full.py --num-samples 2000 --paired-states --seed 42
  python train_dapn_encoder_full.py --load-data artifacts/obs.npz --epochs 50
"""

import os
import sys
from pathlib import Path
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

# ---- Path setup ----
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cyberwheel"))

# ---- Local imports ----
from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_observation_encoder import DAPNObservationTranslator  # keeps your encoder saving/loading
from adapters.unified_full_obs_preprocessor import UnifiedFullObsPreprocessor
from config.env_builders import make_cbs_env, make_cw_env
from adapters.kill_chain import stage_from_cbs, stage_from_cw, KILL_CHAIN_STAGES
import torch.nn.functional as F


# Legacy aliases so rest of file is unchanged
kill_chain_stage_from_cbs = stage_from_cbs
kill_chain_stage_from_cw  = stage_from_cw


# Stage functions live in adapters/kill_chain.py — imported above as aliases.


# =============================================================================
# Full-episode collection (stage-balanced, unlike reset-only)
# =============================================================================
def _collect_full_episodes(env, n_samples, agent=None, is_cbs=False, ep_seed=None, label=""):
    """
    Run full episodes, collecting raw obs at every step so all 5 kill-chain stages
    are represented — not just stage 0/1 seen at reset().
    """
    obs_list = []
    pbar = tqdm(total=n_samples, desc=f"Collecting {label}")
    ep = 0
    while len(obs_list) < n_samples:
        s = (ep_seed + ep) if ep_seed is not None else None
        obs, _ = env.reset(seed=s)
        done = truncated = False
        while not (done or truncated) and len(obs_list) < n_samples:
            raw = getattr(env, "_last_raw_obs", None)
            if raw is None:
                raw = obs if (isinstance(obs, dict) if is_cbs else isinstance(obs, np.ndarray)) else ({} if is_cbs else np.array([], dtype=np.float32))
            obs_list.append(raw)
            if agent is not None:
                if hasattr(agent, "predict"):
                    from gymnasium import spaces as gym_spaces

                    obs_space = getattr(agent, "observation_space", None)
                    # MultiInputPolicy (e.g. CBS UnifiedSecEnv) must receive the full Dict obs
                    if isinstance(obs_space, gym_spaces.Dict) and isinstance(obs, dict):
                        obs_for_pred = obs
                    else:
                        obs_for_pred = obs["obs"] if isinstance(obs, dict) else obs
                    action, _ = agent.predict(obs_for_pred, deterministic=False)
                    action = int(np.asarray(action).squeeze())
                else:
                    obs_for_pred = obs["obs"] if isinstance(obs, dict) else obs
                    # CyberWheel RLPolicy (torch): actor expects flat float vector
                    import torch
                    from torch.distributions import Categorical

                    device = next(agent.parameters()).device
                    x = torch.as_tensor(
                        np.asarray(obs_for_pred, dtype=np.float32).ravel(),
                        device=device,
                    )
                    logits = agent.actor(x)
                    action = int(Categorical(logits=logits).sample().item())
            else:
                action = env.action_space.sample()
            obs, _, done, truncated, _ = env.step(action)
            pbar.update(1)
        ep += 1
    pbar.close()
    return obs_list[:n_samples]


# =============================================================================
# Stage-based pairing (semantic alignment via kill-chain stage)
# =============================================================================
def build_stage_pairs(source_obs_list, target_obs_list):
    """
    Pair each CW observation with a CBS observation at the same kill-chain stage.
    Falls back to the nearest non-empty stage if an exact match is unavailable.

    Returns:
        paired_src: list of CW raw obs
        paired_tgt: list of CBS raw obs (same stage as corresponding paired_src)
        stage_counts: dict showing how many pairs were formed per stage
    """
    # Label every obs with its kill-chain stage
    cw_staged  = [(kill_chain_stage_from_cw(o),  o) for o in source_obs_list]
    cbs_staged = [(kill_chain_stage_from_cbs(o), o) for o in target_obs_list]

    # Build per-stage buckets for both domains
    cw_buckets:  dict[int, list] = {k: [] for k in range(KILL_CHAIN_STAGES)}
    cbs_buckets: dict[int, list] = {k: [] for k in range(KILL_CHAIN_STAGES)}
    for stage, obs in cw_staged:
        cw_buckets[stage].append(obs)
    for stage, obs in cbs_staged:
        cbs_buckets[stage].append(obs)

    paired_src, paired_tgt = [], []
    stage_counts = {k: 0 for k in range(KILL_CHAIN_STAGES)}

    rng = np.random.default_rng(42)
    skipped_cw  = {k: 0 for k in range(KILL_CHAIN_STAGES)}
    skipped_cbs = {k: 0 for k in range(KILL_CHAIN_STAGES)}

    # Pass 1: pair every CW obs with a same-stage CBS obs (CBS sampled w/ replacement)
    for stage, cw_obs in cw_staged:
        if cbs_buckets[stage]:
            tgt = cbs_buckets[stage][int(rng.choice(len(cbs_buckets[stage])))]
            paired_src.append(cw_obs)
            paired_tgt.append(tgt)
            stage_counts[stage] += 1
        else:
            skipped_cw[stage] += 1  # drop — no same-stage CBS partner

    # Pass 2: pair remaining CBS obs that have no CW partner yet (CBS stage 3 surplus)
    # This utilises the CBS observations that Pass 1 never sampled.
    for stage, cbs_obs in cbs_staged:
        if cw_buckets[stage]:
            src = cw_buckets[stage][int(rng.choice(len(cw_buckets[stage])))]
            paired_src.append(src)
            paired_tgt.append(cbs_obs)
            stage_counts[stage] += 1
        else:
            skipped_cbs[stage] += 1  # drop — no same-stage CW partner

    print("Stage pair counts (after both passes):", {k: v for k, v in stage_counts.items() if v > 0})
    dropped_cw  = {k: v for k, v in skipped_cw.items()  if v > 0}
    dropped_cbs = {k: v for k, v in skipped_cbs.items() if v > 0}
    if dropped_cw:
        print(f"  CW obs dropped (no CBS partner at same stage): {dropped_cw}")
    if dropped_cbs:
        print(f"  CBS obs dropped (no CW partner at same stage): {dropped_cbs}")
    return paired_src, paired_tgt, stage_counts


# =============================================================================
# Paired observations dataset (for pair alignment loss)
# =============================================================================
class PairedObservationDataset(Dataset):
    """Dataset of (src_512d, tgt_512d) pairs for MSE alignment loss."""

    def __init__(self, paired_src, paired_tgt, preprocessor, norm_mean, norm_std, clip_z=5.0):
        assert len(paired_src) == len(paired_tgt)
        self.paired_src = paired_src
        self.paired_tgt = paired_tgt
        self.preprocessor = preprocessor
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.clip_z = clip_z

    def _to_512_norm(self, obs):
        if isinstance(obs, dict):
            v = self.preprocessor.preprocess_cbs(obs)
        else:
            v = self.preprocessor.preprocess_cw(obs)
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        d = self.preprocessor.unified_dim
        if v.shape[0] < d:
            v = np.pad(v, (0, d - v.shape[0]))
        elif v.shape[0] > d:
            v = v[:d]
        z = (v - self.norm_mean) / self.norm_std
        if self.clip_z is not None:
            z = np.clip(z, -self.clip_z, self.clip_z)
        return z.astype(np.float32)

    def __len__(self):
        return len(self.paired_src)

    def __getitem__(self, idx):
        src = torch.from_numpy(self._to_512_norm(self.paired_src[idx]))
        tgt = torch.from_numpy(self._to_512_norm(self.paired_tgt[idx]))
        return src, tgt


# =============================================================================
# Balanced batch sampler (ensures each batch has both domains)
# =============================================================================
class BalancedDomainBatchSampler(Sampler):
    """
    Yields batches with equal number of source and target indices.
    Assumes dataset indexing is: [source][target][val].
    """
    def __init__(self, n_source: int, n_target: int, batch_size: int, drop_last: bool = True, seed: int = 0):
        if batch_size % 2 != 0:
            raise ValueError("batch_size must be even for balanced batching")
        self.n_source = n_source
        self.n_target = n_target
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

        self.source_indices = np.arange(0, n_source)
        self.target_indices = np.arange(n_source, n_source + n_target)

    def __iter__(self):
        src = self.source_indices.copy()
        tgt = self.target_indices.copy()
        self.rng.shuffle(src)
        self.rng.shuffle(tgt)

        n_batches = min(len(src), len(tgt)) // self.half
        for b in range(n_batches):
            s = src[b * self.half:(b + 1) * self.half]
            t = tgt[b * self.half:(b + 1) * self.half]
            batch = np.concatenate([s, t])
            self.rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return min(self.n_source, self.n_target) // self.half


# =============================================================================
# Dataset (raw obs -> 512D -> normalize -> tensor)
# =============================================================================
class ObservationDataset(Dataset):
    """
    Stores raw observations and converts them to fixed-size 512D tensors on-the-fly.
    """
    def __init__(self, source_obs_list, target_obs_list, val_obs_list=None,
                 preprocessor=None, norm_mean=None, norm_std=None, clip_z=5.0):
        self.source_obs = source_obs_list
        self.target_obs = target_obs_list
        self.val_obs = val_obs_list or []
        self.total_samples = len(self.source_obs) + len(self.target_obs) + len(self.val_obs)

        if preprocessor is None:
            raise ValueError("preprocessor must be provided")
        self.preprocessor = preprocessor

        # Normalization stats (512D)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.clip_z = clip_z

    def __len__(self):
        return self.total_samples

    def _to_512(self, obs):
        # CBS raw obs is dict
        if isinstance(obs, dict):
            v = self.preprocessor.preprocess_cbs(obs)
        # CW raw obs is numpy array
        elif isinstance(obs, np.ndarray):
            v = self.preprocessor.preprocess_cw(obs)
        # if torch tensor already, return as numpy
        elif isinstance(obs, torch.Tensor):
            v = obs.detach().cpu().numpy()
        else:
            v = np.asarray(obs, dtype=np.float32)

        v = np.asarray(v, dtype=np.float32).reshape(-1)
        d = self.preprocessor.unified_dim
        if v.shape[0] < d:
            v = np.pad(v, (0, d - v.shape[0]), mode="constant")
        elif v.shape[0] > d:
            v = v[:d]
        return v

    def _normalize(self, v512: np.ndarray) -> np.ndarray:
        if self.norm_mean is None or self.norm_std is None:
            # If stats not provided, return raw (not recommended)
            return v512
        z = (v512 - self.norm_mean) / self.norm_std
        if self.clip_z is not None:
            z = np.clip(z, -self.clip_z, self.clip_z)
        return z.astype(np.float32)

    def __getitem__(self, idx):
        if idx < len(self.source_obs):
            obs = self.source_obs[idx]
            label = 0
        elif idx < len(self.source_obs) + len(self.target_obs):
            obs = self.target_obs[idx - len(self.source_obs)]
            label = 1
        else:
            obs = self.val_obs[idx - len(self.source_obs) - len(self.target_obs)]
            label = 2

        v = self._to_512(obs)
        v = self._normalize(v)
        return torch.from_numpy(v), label


# =============================================================================
# Helper: compute dataset normalization stats over 512D vectors
# =============================================================================
def compute_stats(obs_list, preprocessor, max_items=5000):
    n = min(len(obs_list), max_items)
    if n == 0:
        raise ValueError("Cannot compute stats on empty list")

    xs = []
    for i in range(n):
        obs = obs_list[i]
        if isinstance(obs, dict):
            v = preprocessor.preprocess_cbs(obs)
        else:
            v = preprocessor.preprocess_cw(obs)
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        d = preprocessor.unified_dim
        if v.shape[0] < d:
            v = np.pad(v, (0, d - v.shape[0]), mode="constant")
        elif v.shape[0] > d:
            v = v[:d]
        xs.append(v)

    X = np.stack(xs, axis=0)
    mean = X.mean(axis=0).astype(np.float32)
    std = (X.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std


# =============================================================================
# Domain discriminator (logits output for BCEWithLogitsLoss)
# =============================================================================
class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, 1)  # logits
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Observation collection (raw obs)
# =============================================================================
def collect_observations(
    num_samples=1000,
    save_path=None,
    cw_available=True,
    use_3_domains=True,
    cw_agent_path=None,
    cbs_agent_path=None,
    use_paired_states=False,
    seed=None,
):
    """
    Collect FULL RAW observations:
      - Source: Cyberwheel raw obs (numpy array)
      - Target: CBS raw obs (dict)
      - Val: split 20% from target

    Important fix:
      - If use_paired_states=True, we DO NOT step after reset to preserve pairing.
      - Also we force CBS_ENV to CyberBattleCW10-v0 to better match topology.
    """
    print("Collecting FULL RAW observations (stored as dicts for CBS, arrays for CW)")

    if use_paired_states and seed is None:
        seed = 42
        print(f"Using default seed {seed} for paired state collection")

    # Force CBS topology in paired mode
    original_cbs_env = None
    if use_paired_states:
        original_cbs_env = os.environ.get("CBS_ENV")
        os.environ["CBS_ENV"] = "CyberBattleCW10-v0"
        print("Paired mode: set CBS_ENV=CyberBattleCW10-v0 to match Cyberwheel topology")

    # Optional agent loading (kept, but paired mode won't step anyway)
    cw_agent = None
    cbs_agent = None

    if cw_agent_path and cw_agent_path != "path/to/cyberwheel_agent.pt" and os.path.exists(cw_agent_path):
        try:
            from cyberwheel.utils import RLPolicy
            from eval.eval_cw_checkpoints_on_cbs import infer_cyberwheel_config
            print(f"Loading Cyberwheel agent from {cw_agent_path}...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            action_space_size, obs_space_shape = infer_cyberwheel_config(cw_agent_path)
            cw_agent = RLPolicy(action_space_shape=action_space_size, obs_space_shape=obs_space_shape).to(device)
            state_dict = torch.load(cw_agent_path, map_location=device)
            cw_agent.load_state_dict(state_dict)
            cw_agent.eval()
            print(f"✓ Loaded Cyberwheel agent")
        except Exception as e:
            print(f"Warning: could not load Cyberwheel agent: {e}")
            cw_agent = None

    if cbs_agent_path and cbs_agent_path != "path/to/cbs_agent.zip" and os.path.exists(cbs_agent_path):
        try:
            from stable_baselines3 import PPO
            print(f"Loading CBS agent from {cbs_agent_path}...")
            cbs_agent = PPO.load(cbs_agent_path)
            print("✓ Loaded CBS agent")
        except Exception as e:
            print(f"Warning: could not load CBS agent: {e}")
            cbs_agent = None

    # ---- Source domain (CW) — full-episode rollouts ----
    source_obs_list = []
    source_actions_list = []
    if cw_available:
        print("Collecting source (Cyberwheel) full-episode observations...")
        try:
            cw_env = UnifiedSecEnv("cw", cw_factory=make_cw_env)
            source_obs_list = _collect_full_episodes(
                cw_env, num_samples, agent=cw_agent, is_cbs=False,
                ep_seed=seed, label="CW"
            )
            source_actions_list = [0] * len(source_obs_list)  # actions not needed for DANN
        except Exception as e:
            print(f"Warning: could not collect Cyberwheel obs: {e}")
            print("Falling back to CBS as source domain...")
            cbs_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
            source_obs_list = _collect_full_episodes(
                cbs_env, num_samples, agent=cbs_agent, is_cbs=True,
                ep_seed=seed, label="CW-fallback"
            )
            source_actions_list = [0] * len(source_obs_list)
    else:
        print("Skipping source (Cyberwheel)")

    # ---- Target domain (CBS) — full-episode rollouts ----
    print("Collecting target (CyberBattleSim) full-episode observations...")
    target_cbs_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    target_obs_list = _collect_full_episodes(
        target_cbs_env, num_samples, agent=cbs_agent, is_cbs=True,
        ep_seed=seed, label="CBS"
    )
    target_actions_list = [0] * len(target_obs_list)

    # ---- Validation split ----
    val_obs_list = []
    if use_3_domains and len(target_obs_list) > 0:
        split_idx = len(target_obs_list) // 5  # 20%
        val_obs_list = target_obs_list[:split_idx]
        target_obs_list = target_obs_list[split_idx:]
        target_actions_list = target_actions_list[split_idx:]
        print(f"Validation split: {len(val_obs_list)}; Target train: {len(target_obs_list)}")

    # Ensure action lengths match obs lengths
    source_actions_list = source_actions_list[:len(source_obs_list)]
    target_actions_list = target_actions_list[:len(target_obs_list)]

    # Shuffle
    if use_paired_states and len(source_obs_list) > 0 and len(target_obs_list) > 0:
        min_len = min(len(source_obs_list), len(target_obs_list))
        idxs = list(range(min_len))
        random.shuffle(idxs)
        source_obs_list = [source_obs_list[i] for i in idxs] + source_obs_list[min_len:]
        source_actions_list = [source_actions_list[i] for i in idxs] + source_actions_list[min_len:]
        target_obs_list = [target_obs_list[i] for i in idxs] + target_obs_list[min_len:]
        target_actions_list = [target_actions_list[i] for i in idxs] + target_actions_list[min_len:]
        print(f"Shuffled paired samples together ({min_len})")
    else:
        if len(source_obs_list) > 0:
            idxs = list(range(len(source_obs_list)))
            random.shuffle(idxs)
            source_obs_list = [source_obs_list[i] for i in idxs]
            source_actions_list = [source_actions_list[i] for i in idxs]
        if len(target_obs_list) > 0:
            idxs = list(range(len(target_obs_list)))
            random.shuffle(idxs)
            target_obs_list = [target_obs_list[i] for i in idxs]
            target_actions_list = [target_actions_list[i] for i in idxs]

    # Save
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        np.savez(
            save_path,
            source_obs=source_obs_list,
            source_actions=source_actions_list,
            target_obs=target_obs_list,
            target_actions=target_actions_list,
            val_obs=val_obs_list,
        )
        print(f"Saved observations to {save_path}")

    # Restore env var
    if original_cbs_env is not None:
        if original_cbs_env:
            os.environ["CBS_ENV"] = original_cbs_env
        else:
            os.environ.pop("CBS_ENV", None)

    return source_obs_list, target_obs_list, val_obs_list, source_actions_list, target_actions_list


# =============================================================================
# Training: Proper DANN + balanced batches + dataset normalization
# =============================================================================
def train_dapn_encoder(
    source_obs_list,
    target_obs_list,
    val_obs_list=None,
    feature_size=256,
    num_epochs=50,
    batch_size=64,
    learning_rate=1e-3,
    device=None,
    save_path="artifacts/transfer_models/dapn_encoder.pt",
    lambda_adv=0.5,
    lambda_pair=0.1,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    if len(source_obs_list) == 0 or len(target_obs_list) == 0:
        raise ValueError("Need both source and target samples for domain alignment")

    val_obs_list = val_obs_list or []

    # Preprocessor for full observations
    preprocessor = UnifiedFullObsPreprocessor(unified_dim=512)

    # Compute dataset normalization stats on 512D vectors
    print("Computing normalization stats (mean/std) on 512D unified vectors...")
    mean_s, std_s = compute_stats(source_obs_list, preprocessor)
    mean_t, std_t = compute_stats(target_obs_list, preprocessor)
    norm_mean = 0.5 * (mean_s + mean_t)
    norm_std = 0.5 * (std_s + std_t)

    # Translator uses encoders (512 -> feature_size)
    use_shared = os.environ.get("DAPN_USE_SHARED_ENCODER", "1") == "1"
    translator = DAPNObservationTranslator(
        use_dapn=True,
        feature_size=feature_size,
        input_dim=512,
        device=device,
        use_adversarial=True,  # we will override domain_adapter with logits discriminator below
    )

    # Put encoders in train mode
    if getattr(translator, "use_shared_encoder", False):
        translator.shared_encoder.train()
        encoder_params = list(translator.shared_encoder.parameters())
    else:
        translator.cbs_encoder.train()
        if translator.cw_encoder is not None:
            translator.cw_encoder.train()
            encoder_params = list(translator.cbs_encoder.parameters()) + list(translator.cw_encoder.parameters())
        else:
            encoder_params = list(translator.cbs_encoder.parameters())

    # Override domain adapter with a stable logits discriminator
    translator.domain_adapter = DomainDiscriminator(feature_dim=feature_size, hidden=256).to(device)
    translator.domain_adapter.train()

    # Dataset + balanced batches (source/target only; val is kept but not batched here)
    dataset = ObservationDataset(
        source_obs_list,
        target_obs_list,
        val_obs_list=val_obs_list,
        preprocessor=preprocessor,
        norm_mean=norm_mean,
        norm_std=norm_std,
        clip_z=5.0,
    )

    sampler = BalancedDomainBatchSampler(
        n_source=len(source_obs_list),
        n_target=len(target_obs_list),
        batch_size=batch_size,
        drop_last=True,
        seed=42,
    )
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    # Build stage-paired dataset for pair alignment loss
    print("Building kill-chain-stage paired observations...")
    paired_src, paired_tgt, stage_counts = build_stage_pairs(source_obs_list, target_obs_list)
    pair_loader = None
    if len(paired_src) > 0 and lambda_pair > 0:
        pair_dataset = PairedObservationDataset(
            paired_src, paired_tgt, preprocessor, norm_mean, norm_std, clip_z=5.0
        )
        pair_loader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True,
                                 drop_last=True, num_workers=0)
        print(f"  Pair dataset size: {len(pair_dataset)} (lambda_pair={lambda_pair})")
    else:
        print("  Skipping pair loss (no pairs or lambda_pair=0)")

    # Optimizers
    optimizer_encoder = optim.Adam(encoder_params, lr=learning_rate)
    optimizer_disc = optim.Adam(translator.domain_adapter.parameters(), lr=learning_rate * 0.1)

    # Loss (logits-based)
    bce_logits = nn.BCEWithLogitsLoss()

    print("\n" + "=" * 80)
    print("DANN Training")
    print(f"  Source samples: {len(source_obs_list)}")
    print(f"  Target samples: {len(target_obs_list)}")
    print(f"  Val samples:    {len(val_obs_list)} (not used in loss)")
    print(f"  Input: 512D unified vectors  ->  {feature_size}D features")
    print(f"  Batch: {batch_size} (balanced)")
    print(f"  lr encoder: {learning_rate} | lr disc: {learning_rate * 0.1}")
    print(f"  lambda_adv: {lambda_adv}  lambda_pair: {lambda_pair}")
    print("=" * 80 + "\n")

    pair_iter = iter(pair_loader) if pair_loader is not None else None

    for epoch in range(num_epochs):
        translator.domain_adapter.train()
        if getattr(translator, "use_shared_encoder", False):
            translator.shared_encoder.train()
        else:
            translator.cbs_encoder.train()
            if translator.cw_encoder is not None:
                translator.cw_encoder.train()

        total_enc_loss = 0.0
        total_disc_loss = 0.0
        total_pair_loss = 0.0
        n_batches = 0
        # Reset pair iterator each epoch
        if pair_loader is not None:
            pair_iter = iter(pair_loader)

        for obs_batch, domain_labels in dataloader:
            # Balanced sampler gives only 0 and 1 labels in practice
            obs_batch = obs_batch.to(device)
            domain_labels = domain_labels.to(device)

            source_mask = (domain_labels == 0)
            target_mask = (domain_labels == 1)
            if not (source_mask.any() and target_mask.any()):
                continue

            # Encode
            if getattr(translator, "use_shared_encoder", False):
                source_feat = translator.shared_encoder(obs_batch[source_mask])
                target_feat = translator.shared_encoder(obs_batch[target_mask])
            else:
                # CW uses cw_encoder, CBS uses cbs_encoder
                if translator.cw_encoder is None:
                    # fallback: all use cbs encoder
                    source_feat = translator.cbs_encoder(obs_batch[source_mask])
                else:
                    source_feat = translator.cw_encoder(obs_batch[source_mask])
                target_feat = translator.cbs_encoder(obs_batch[target_mask])

            feats = torch.cat([source_feat, target_feat], dim=0)

            # Domain targets: 0 for source, 1 for target
            y_dom = torch.cat(
                [
                    torch.zeros(source_feat.size(0), 1, device=device),
                    torch.ones(target_feat.size(0), 1, device=device),
                ],
                dim=0,
            )

            # -----------------------------
            # (1) Update discriminator
            # -----------------------------
            optimizer_disc.zero_grad()
            logits_detached = translator.domain_adapter(feats.detach())
            disc_loss = bce_logits(logits_detached, y_dom)
            disc_loss.backward()
            optimizer_disc.step()

            # -----------------------------
            # (2) Update encoder to confuse discriminator
            #     Encoder wants discriminator to fail -> maximize disc loss
            #     Implemented by minimizing (-disc_loss).
            # -----------------------------
            optimizer_encoder.zero_grad()
            logits = translator.domain_adapter(feats)  # no detach, grads flow to encoder
            adv_loss = bce_logits(logits, y_dom)
            enc_loss = -lambda_adv * adv_loss

            # Pair alignment loss: same kill-chain stage → same feature space
            pair_loss_val = torch.tensor(0.0, device=device)
            if pair_loader is not None and lambda_pair > 0:
                try:
                    src_p, tgt_p = next(pair_iter)
                except StopIteration:
                    pair_iter = iter(pair_loader)
                    src_p, tgt_p = next(pair_iter)
                src_p = src_p.to(device)
                tgt_p = tgt_p.to(device)
                if getattr(translator, "use_shared_encoder", False):
                    src_feat_p = translator.shared_encoder(src_p)
                    tgt_feat_p = translator.shared_encoder(tgt_p)
                else:
                    enc_src = translator.cw_encoder if translator.cw_encoder is not None else translator.cbs_encoder
                    src_feat_p = enc_src(src_p)
                    tgt_feat_p = translator.cbs_encoder(tgt_p)
                pair_loss_val = F.mse_loss(src_feat_p, tgt_feat_p)
                enc_loss = enc_loss + lambda_pair * pair_loss_val

            enc_loss.backward()
            optimizer_encoder.step()

            total_disc_loss += disc_loss.item()
            total_enc_loss += enc_loss.item()
            total_pair_loss += pair_loss_val.item()
            n_batches += 1

        if n_batches == 0:
            print(f"Epoch {epoch+1}: no valid batches (check sampler / data).")
            continue

        avg_disc = total_disc_loss / n_batches
        avg_enc = total_enc_loss / n_batches
        avg_pair = total_pair_loss / n_batches

        print_freq = 1 if num_epochs <= 20 else 5
        if (epoch + 1) % print_freq == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>3}/{num_epochs}: disc_loss={avg_disc:.4f} | enc_loss={avg_enc:.4f} | pair_loss={avg_pair:.4f}")

    # Save encoder + normalization stats so inference uses identical preprocessing
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    translator.save_encoder(save_path)
    # Patch norm stats into checkpoint so DAPNUnifiedFullObsTranslator can load them
    import torch as _torch
    _ckpt = _torch.load(save_path, map_location="cpu", weights_only=False)
    _ckpt["norm_mean"] = norm_mean   # shape (512,)
    _ckpt["norm_std"]  = norm_std    # shape (512,)
    _ckpt["clip_z"]    = 5.0
    _torch.save(_ckpt, save_path)
    print(f"\n✓ Saved trained encoder (+ norm stats) to {save_path}")
    return translator


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DAPN encoder (full obs, balanced DANN)")
    parser.add_argument("--num-samples", type=int, default=1000, help="Samples per domain")
    parser.add_argument("--feature-size", type=int, default=256, help="Feature dim")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (even number)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--lambda-adv", type=float, default=0.5, help="Adversarial loss weight")
    parser.add_argument("--load-data", type=str, default=None, help="Load obs .npz")
    parser.add_argument("--save-data", type=str, default=None, help="Save obs .npz")
    parser.add_argument("--save-encoder", type=str, default="artifacts/transfer_models/dapn_encoder.pt")
    parser.add_argument("--cbs-only", action="store_true", help="Skip Cyberwheel")
    parser.add_argument("--cw-agent", type=str, default=None, help="Cyberwheel agent path")
    parser.add_argument("--cbs-agent", type=str, default=None, help="CBS agent path")
    parser.add_argument("--paired-states", action="store_true", help="Pair resets by seed and use CW topology in CBS")
    parser.add_argument("--seed", type=int, default=None, help="Seed base for paired states")
    parser.add_argument("--lambda-pair", type=float, default=0.1, help="Stage-pair alignment loss weight")

    args = parser.parse_args()

    # Load or collect
    if args.load_data and os.path.exists(args.load_data):
        print(f"Loading observations from {args.load_data}")
        data = np.load(args.load_data, allow_pickle=True)
        source_obs_list = data["source_obs"].tolist() if "source_obs" in data else []
        target_obs_list = data["target_obs"].tolist() if "target_obs" in data else []
        val_obs_list = data["val_obs"].tolist() if "val_obs" in data else []
        source_actions_list = data["source_actions"].tolist() if "source_actions" in data else None
        target_actions_list = data["target_actions"].tolist() if "target_actions" in data else None
    else:
        source_obs_list, target_obs_list, val_obs_list, source_actions_list, target_actions_list = collect_observations(
            num_samples=args.num_samples,
            save_path=args.save_data,
            cw_available=not args.cbs_only,
            use_3_domains=not args.cbs_only,
            cw_agent_path=args.cw_agent,
            cbs_agent_path=args.cbs_agent,
            use_paired_states=args.paired_states,
            seed=args.seed,
        )

    if len(source_obs_list) == 0:
        print("Error: No source observations collected.")
        sys.exit(1)
    if len(target_obs_list) == 0:
        print("Error: No target observations collected.")
        sys.exit(1)

    translator = train_dapn_encoder(
        source_obs_list=source_obs_list,
        target_obs_list=target_obs_list,
        val_obs_list=val_obs_list,
        feature_size=args.feature_size,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_encoder,
        lambda_adv=args.lambda_adv,
        lambda_pair=args.lambda_pair,
    )

    print("\n✓ Training complete!")
