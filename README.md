Last updated on 10/19/2025 (not maintained)
## RL Transfer: CyberBattleSim → Cyberwheel

Train an RL policy on Microsoft CyberBattleSim (CBS) and zero-shot test it on Cyberwheel (CW) using a unified adapter layer for observations, actions, and rewards.

### Key Idea
- Unified Gym-like wrapper exposes both envs under the same API.
- Observations are mapped to a strict 8D normalized vector with consistent semantics.
- Actions are mapped from a small semantic set to each backend.
- Rewards are clamped to [-1, 1] for stability and comparability.


## Prerequisites
- Python 3.10 available on PATH
- Poetry for Cyberwheel (poetry >= 1.5)

### 1) CyberBattleSim (root venv)
Setup once:
```bash
cd /home/ssaika/rl-transfer-sec
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e CyberBattleSim
pip install gymnasium==0.29.1 stable-baselines3==2.3.2 numpy==1.26.4 pandas==2.2.2 \
  torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```
Train a tiny CBS policy and run a quick demo:

PYTHONPATH=/home/ssaika/rl-transfer-sec python3 train/train_cbs_ppo_minimal.py
PYTHONPATH=/home/ssaika/rl-transfer-sec python3 test_transfer_simple.py
```
### 2) Cyberwheel (Poetry env)
Setup once:
```bash
cd /home/ssaika/rl-transfer-sec/cyberwheel
poetry install
poetry run python3 -m cyberwheel -h
```

Minimal RL training:
```bash
cd /home/ssaika/rl-transfer-sec/cyberwheel
poetry run python3 -m cyberwheel train train_rl_red_agent_vs_rl_blue.yaml \
  --experiment-name QuickCheck \
  --debug-mode True --num-envs 1 --num-steps 128 --total-timesteps 128 --num-saves 1 \
  --learning-rate 1e-4 --ent-coef 0.0 --clip-coef 0.1 --norm-adv False --update-epochs 1 --num-minibatches 1
```

Note: When `--debug-mode True` is used, the experiment name is prefixed with `DEBUG_`.

Evaluate the saved checkpoint:
```bash
poetry run python3 -m cyberwheel evaluate evaluate_rl_red_vs_rl_blue.yaml \
  --experiment-name DEBUG_QuickCheck --checkpoint agent
```

## Zero-Shot Transfer (CBS → Cyberwheel)
This script loads the CBS-trained policy and runs it first on CBS, then attempts CW using the adapters.
```bash
cd /home/ssaika/rl-transfer-sec
PYTHONPATH=/home/ssaika/rl-transfer-sec python3 test_transfer_simple.py

source .venv_cw/bin/activate
python -m cyberwheel train train_red.yaml --track False --num-envs 1 --total-timesteps 3000
```



