#!/usr/bin/env python3
"""
Diagnostic script to identify Cyberwheel issues
"""

import os
import sys
from pathlib import Path

print("=" * 60)
print("Cyberwheel Diagnostic Tool")
print("=" * 60)

# Check 1: Python path
print("\n1. Checking Python path...")
project_root = Path(__file__).parent
cyberwheel_path = project_root / "cyberwheel"
print(f"   Project root: {project_root}")
print(f"   Cyberwheel path: {cyberwheel_path}")
print(f"   Cyberwheel exists: {cyberwheel_path.exists()}")

if cyberwheel_path.exists():
    sys.path.insert(0, str(cyberwheel_path))
    print("   ✓ Added cyberwheel to path")
else:
    print("   ✗ Cyberwheel directory not found!")

sys.path.insert(0, str(project_root))
print(f"   ✓ Added project root to path")

# Check 2: Required files
print("\n2. Checking required files...")
files_to_check = [
    "cyberwheel/cyberwheel/data/configs/environment/credential_preference_scenario.yaml",
    "cyberwheel/cyberwheel/data/configs/network/credential-preference-100host-network.yaml",
    "cyberwheel/cyberwheel/data/configs/red_agent/rl_red_agent_credential_preference.yaml",
]

for file_path in files_to_check:
    full_path = project_root / file_path
    exists = full_path.exists()
    status = "✓" if exists else "✗"
    print(f"   {status} {file_path}: {exists}")

# Check 3: Environment variables
print("\n3. Checking environment variables...")
cw_env = os.environ.get("CW_ENV_YAML", "NOT SET")
print(f"   CW_ENV_YAML: {cw_env}")

# Check 4: Try importing
print("\n4. Testing imports...")
try:
    import cyberwheel
    print("   ✓ cyberwheel module imported")
except Exception as e:
    print(f"   ✗ Failed to import cyberwheel: {e}")

try:
    from cyberwheel.network.network_base import Network
    print("   ✓ cyberwheel.network.network_base imported")
except Exception as e:
    print(f"   ✗ Failed to import network_base: {e}")

try:
    from adapters.unified_env import UnifiedSecEnv
    print("   ✓ adapters.unified_env imported")
except Exception as e:
    print(f"   ✗ Failed to import unified_env: {e}")

try:
    from config.env_builders import make_cw_env
    print("   ✓ config.env_builders imported")
except Exception as e:
    print(f"   ✗ Failed to import env_builders: {e}")

# Check 5: Try creating environment
print("\n5. Testing environment creation...")
try:
    os.environ["CW_ENV_YAML"] = "credential_preference_scenario.yaml"
    from adapters.unified_env import UnifiedSecEnv
    from config.env_builders import make_cw_env
    
    print("   Creating environment...")
    env = UnifiedSecEnv("cw", cw_factory=make_cw_env)
    print("   ✓ Environment created successfully!")
    
    print("   Resetting environment...")
    obs, info = env.reset()
    print(f"   ✓ Environment reset successful!")
    print(f"   Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    print(f"   Action space: {env.action_space}")
    
except Exception as e:
    print(f"   ✗ Failed to create environment: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Diagnostic complete!")
print("=" * 60)

