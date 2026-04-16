RL Transfer Learning: Cyberwheel → CyberBattleSim

**CyberBattleSim** using DAPN for domain adaptation.
## Prerequisites

- **Python 3.10**
- **Poetry** (>= 1.5)
- **Graphviz**

### GPU (optional)

All **training** (PPO and DAPN encoder) uses GPU when available. Install PyTorch with CUDA for GPU, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu118`. To force CPU, set `USE_GPU=0` before running. Data collection scripts (`collect_lockstep_reward_aligned.py`, `run_paired_rollout.py`) are CPU-only (env simulators do not run on GPU).

Setup Project
```bash
# Navigate to project directory
cd /home/ssaika/rl-transfer-sec-clean

# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install CyberBattleSim
pip install -e CyberBattleSim
pip install gymnasium==0.29.1 stable-baselines3==2.3.2 numpy==1.26.4 pandas==2.2.2
pip install tqdm pydantic jsonpickle python-dotenv networkx pyyaml
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Set up Cyberwheel
cd cyberwheel
poetry install
cd ..

# Set environment variables
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
export CW_ENV_YAML=credential_preference_scenario.yaml
```

## Run the Project

```bash
# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
export CW_ENV_YAML=credential_preference_scenario.yaml

# Run the complete pipeline
bash run_transfer_learning.sh
```

This will:

1. Train DAPN encoder (if needed)
2. Train PPO agent on Cyberwheel
3. Evaluate transfer to CyberBattleSim

## Output Files

After running:

- `artifacts/transfer_models/dapn_encoder.pt` - Trained encoder
- `artifacts/policies/cw_ppo_dapn.zip` - Trained policy



## Encoder full training (policy-based collection + episodic encoder)

Train CW and CBS policies first, then collect data with those policies and train the episodic DAPN encoder:

```bash
source .venv/bin/activate
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
export CW_ENV_YAML=credential_preference_scenario.yaml

# 1. Train Cyberwheel policy
python train/train_cw_ppo_very_short.py

# 2. Train CyberBattleSim policy
python train/train_cbs_ppo_very_short.py

# 3. Collect data with both policies and train the encoder
python train_dapn_encoder_episodic.py \
  --num-samples 2000 \
  --cw-policy artifacts/policies/cw_ppo_very_short.zip \
  --cbs-policy artifacts/policies/cbs_ppo_very_short.zip \
  --max-steps 200 \
  --save-data artifacts/training_data/obs_policy.npz \
  --label-mode situation_action
```

### Deterministic data collection (raw obs only; no encoder)

To collect paired raw observations from CW and CBS (same seed per episode, same action each step), use the standalone script. **No encoder and no policy**—just deterministic round-robin actions. Use the saved data later to train the encoder.

```bash
python collect_deterministic_lockstep.py --num-samples 500 --save artifacts/training_data/lockstep.npz --seed 42 --max-steps 200
```

Then train the encoder on that data: `train_dapn_encoder_episodic.py --load-data artifacts/training_data/lockstep.npz ...`. See `HOW_DATA_COLLECTION_WORKS.md` for details.

### Reward-aligned lockstep (same state → same action → same reward ⇒ same next state)

When both environments share the **same network structure**, you can collect data and use **reward equality** as a check that the next states are aligned: if CW and CBS start from equivalent states (S1, S2), take the same action a1, and get the **same reward**, then the resulting states (S3, S4) are treated as equivalent. The scenario runs until **both** envs reach a goal (e.g. 60% of hosts captured) or max steps.

```bash
# Collect until 60% in both; keep only steps where reward_cw == reward_cbs
python collect_lockstep_reward_aligned.py --goal-pct 0.6 --reward-match-only --save artifacts/training_data/reward_aligned.npz

# Collect with max steps; allow all steps (reward_match flag saved per step)
python collect_lockstep_reward_aligned.py --max-steps 500 --save out.npz --reward-tol 0.01
```

Output includes `reward_cw`, `reward_cbs`, `reward_match`, and `owned_pct_cw` / `owned_pct_cbs` per step. Raw reward is exposed in `info["raw_reward"]` from `UnifiedSecEnv.step()` for comparison.

### Deterministic paired rollout (state-signature assertion)

For **fully deterministic** paired rollouts with **canonical state equivalence** (not just reward):

1. **Seed everything once** at script start: `adapters.deterministic_seed.set_all_seeds(SEED)` (Python, NumPy, Torch).
2. **Same canonical action** in both envs: unified action index 0–6; `DETERMINISTIC_BACKEND_ACTION=1` so the first valid backend action is chosen (no random tie-breaking).
3. **Compare state signatures**: canonical state = (compromised hosts, discovered hosts, creds, progress). Use `UnifiedSecEnv.get_canonical_state(raw_obs)` and a stable hash; assert equality each step.
4. **Run paired** with `run_paired_rollout.py`: asserts initial (optional) and per-step state alignment; stops on mismatch or when progress ≥ goal (e.g. 60%).

```bash
# Strict: assert initial and per-step state alignment (same topology + compatible initial seeding required)
python run_paired_rollout.py --seed 12345 --goal-pct 0.6 --max-steps 2000

# Skip initial assertion if CBS/CW initial seeding differs
python run_paired_rollout.py --no-assert-initial --max-steps 500

# Save dataset
python run_paired_rollout.py --save artifacts/paired_rollout.json
```

Requires both envs to use the **same network structure** (e.g. 10-host from same YAML) so host IDs and semantics match. If a step produces different canonical states, the script raises (or logs with `--continue-on-mismatch`).


