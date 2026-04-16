# Fix for Cyberwheel "Failed" Error

## Quick Fix - Run This:

```bash
cd /home/ssaika/rl-transfer-sec-clean
./run_cyberwheel_test.sh
```

OR manually:

```bash
cd /home/ssaika/rl-transfer-sec-clean
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
export CW_ENV_YAML=credential_preference_scenario.yaml
python test_cyberwheel_only.py
```

## Why It Might Be Failing

The most common reason is that **PYTHONPATH is not set**. The cyberwheel module needs to be in your Python path.

## Permanent Fix

Add this to your `~/.bashrc` or `~/.zshrc`:

```bash
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
```

Then reload your shell:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

## Test Commands

### Test 1: Simple test (shows detailed errors)
```bash
python test_cyberwheel_only.py
```

### Test 2: Full test (both scenarios)
```bash
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
python test_scenarios.py
```

### Test 3: Using the fix script
```bash
./run_cyberwheel_test.sh
```

## If It Still Fails

1. **Check the exact error message:**
   ```bash
   python test_cyberwheel_only.py 2>&1 | tee error.log
   cat error.log
   ```

2. **Verify files exist:**
   ```bash
   ls -la cyberwheel/cyberwheel/data/configs/environment/credential_preference_scenario.yaml
   ls -la cyberwheel/cyberwheel/data/configs/red_agent/rl_red_agent_credential_preference.yaml
   ```

3. **Check Python environment:**
   ```bash
   which python
   python --version
   ```

4. **Run diagnostic:**
   ```bash
   python diagnose_cyberwheel.py
   ```

## Common Errors and Solutions

### Error: "No module named 'cyberwheel'"
**Solution:**
```bash
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
```

### Error: "KeyError: 'host1'"
**Solution:** Already fixed - make sure you're using `credential_preference_scenario.yaml` which uses `rl_red_agent_credential_preference.yaml`

### Error: "FileNotFoundError"
**Solution:** Make sure you're in the project directory:
```bash
cd /home/ssaika/rl-transfer-sec-clean
```

