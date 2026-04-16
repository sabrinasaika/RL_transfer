# Why Rewards Are Constant: Explanation

## Problem

During transfer learning evaluation, all episodes show **exactly 22.00 reward** with **exactly 100 steps**, indicating the agent is stuck in a deterministic behavior pattern.

## Root Cause

### 1. **Insufficient Training**
- The agent was trained for only **1000 steps** (set in `run_transfer_learning.sh`)
- This is far too short for the agent to learn meaningful behavior
- PPO typically needs 10,000-100,000+ steps to learn effective policies

### 2. **Deterministic Action Selection**
- Evaluation uses `deterministic=True` in `model.predict()`
- The deterministic policy learned to **always pick action 6 (impact)**
- Action 6 (impact) is a terminal action that requires:
  - Owning at least one node
  - Having privilege escalation completed
  - These prerequisites are not met in the initial state

### 3. **Invalid Action Sequence**
- The agent tries to execute "impact" immediately without:
  - `ping_sweep` (action 1) - discover network
  - `port_scan` (action 2) - scan for services
  - `discovery` (action 3) - discover vulnerabilities
  - `lateral_move` (action 4) - move to other nodes
  - `privilege_escalation` (action 5) - escalate privileges
  - Only then can `impact` (action 6) succeed

### 4. **Reward Pattern**
- **Step 1**: Gets 22.00 reward (likely initialization bonus or first-action reward)
- **Steps 2-100**: Gets 0.00 reward (invalid action, no progress)
- **Total**: 22.00 / 100 steps = 0.22 average reward per step

## Evidence

From diagnostic script (`diagnose_constant_reward.py`):
```
Step   1: reward= 22.00, action=6 (impact)
Step   2: reward=  0.00, action=6 (impact)
Step   3: reward=  0.00, action=6 (impact)
...
All actions are identical: action 6 (impact)
```

## Solutions

### Option 1: Train for Longer (Recommended)
```bash
# In run_transfer_learning.sh, change:
export CW_TRAIN_STEPS=50000  # Instead of 1000
```

### Option 2: Use Stochastic Actions During Evaluation
```python
# In evaluate_transfer.py, change:
action, _ = model.predict(obs_for_pred, deterministic=False)  # Instead of True
```

This allows the agent to explore different actions and may reveal if it learned anything useful.

### Option 3: Check Training Progress
- Monitor training metrics (episode rewards, policy loss)
- Ensure rewards are increasing during training
- Verify the agent is learning to take valid action sequences

## Expected Behavior After Fix

With proper training:
- Episode rewards should **vary** between episodes
- Agent should learn to take valid action sequences
- Rewards should increase as agent makes progress
- Episode lengths should vary (some episodes end early on success)

## Current Status

The transfer learning **infrastructure works correctly**:
- DAPN encoder loads successfully
- Environment wraps correctly
- Observation spaces match
- Action translation works

The issue is purely that the **agent hasn't learned a good policy** due to insufficient training.
