#!/usr/bin/env python3
"""
Test ONLY Cyberwheel scenario with detailed error reporting
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add cyberwheel to path if it exists
cyberwheel_path = project_root / "cyberwheel"
if cyberwheel_path.exists():
    sys.path.insert(0, str(cyberwheel_path))
    print(f"✓ Added cyberwheel to path: {cyberwheel_path}")
else:
    print(f"✗ Cyberwheel path not found: {cyberwheel_path}")

print("=" * 60)
print("Testing Cyberwheel Credential Preference Scenario")
print("=" * 60)
print(f"Python: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Project root: {project_root}")
print()

try:
    # Set environment variable
    os.environ["CW_ENV_YAML"] = "credential_preference_scenario.yaml"
    print(f"✓ Set CW_ENV_YAML={os.environ['CW_ENV_YAML']}")
    
    # Try importing
    print("\nStep 1: Importing modules...")
    from adapters.unified_env import UnifiedSecEnv
    print("  ✓ Imported UnifiedSecEnv")
    
    from config.env_builders import make_cw_env
    print("  ✓ Imported make_cw_env")
    
    # Create environment
    print("\nStep 2: Creating environment...")
    env = UnifiedSecEnv("cw", cw_factory=make_cw_env)
    print("  ✓ Environment created")
    
    # Reset environment
    print("\nStep 3: Resetting environment...")
    obs, info = env.reset()
    print("  ✓ Environment reset")
    
    print("\n" + "=" * 60)
    print("SUCCESS! Environment Details:")
    print("=" * 60)
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    
    # Test a few steps
    print("\nStep 4: Testing steps...")
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.3f}, done={done}, truncated={truncated}")
        if done or truncated:
            obs, info = env.reset()
    
    print("\n" + "=" * 60)
    print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
    print("=" * 60)
    
except ImportError as e:
    print("\n" + "=" * 60)
    print("✗✗✗ IMPORT ERROR ✗✗✗")
    print("=" * 60)
    print(f"Error: {e}")
    print("\nTrying to fix...")
    print(f"Current sys.path: {sys.path[:3]}")
    print("\nTry running:")
    print("  export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH")
    print("  python test_cyberwheel_only.py")
    import traceback
    traceback.print_exc()
    sys.exit(1)
    
except KeyError as e:
    print("\n" + "=" * 60)
    print("✗✗✗ KEY ERROR ✗✗✗")
    print("=" * 60)
    print(f"Error: {e}")
    print("\nThis usually means a host name is missing from the network.")
    print("Check that the red agent config uses valid host names.")
    import traceback
    traceback.print_exc()
    sys.exit(1)
    
except Exception as e:
    print("\n" + "=" * 60)
    print("✗✗✗ ERROR ✗✗✗")
    print("=" * 60)
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print("\nFull traceback:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

