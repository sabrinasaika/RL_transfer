#!/usr/bin/env python3
"""
Test script for observation transfer implementation.
Tests all components: encoder, dynamics model, integration with ObservationTranslator.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_encoder():
    """Test ObservationEncoder"""
    print("=" * 60)
    print("TEST 1: ObservationEncoder")
    print("=" * 60)
    
    from adapters.transfer_encoder import ObservationEncoder
    
    # Create encoder
    encoder = ObservationEncoder(input_dim=8, feature_size=64)
    print(f"✓ Encoder created: input_dim=8, feature_size=64")
    
    # Test forward pass
    obs = torch.randn(5, 8)  # Batch of 5 observations
    features = encoder(obs)
    
    assert features.shape == (5, 64), f"Expected shape (5, 64), got {features.shape}"
    print(f"✓ Forward pass works: input {obs.shape} -> output {features.shape}")
    
    # Test L2 normalization
    norms = torch.norm(features, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Features should be L2 normalized"
    print(f"✓ Features are L2 normalized (norms: {norms.mean().item():.4f})")
    
    # Test single observation
    obs_single = torch.randn(8)
    features_single = encoder(obs_single)
    assert features_single.shape == (64,), f"Expected shape (64,), got {features_single.shape}"
    print(f"✓ Single observation works: input {obs_single.shape} -> output {features_single.shape}")
    
    print("✓ ObservationEncoder: ALL TESTS PASSED\n")
    return encoder


def test_dynamics_model():
    """Test DynamicsModel"""
    print("=" * 60)
    print("TEST 2: DynamicsModel")
    print("=" * 60)
    
    from adapters.transfer_encoder import DynamicsModel
    
    # Create dynamics model
    dynamics_model = DynamicsModel(feature_size=64, num_actions=7)
    print(f"✓ DynamicsModel created: feature_size=64, num_actions=7")
    
    # Test forward pass
    batch_size = 5
    state_features = torch.randn(batch_size, 64)
    actions = torch.randint(0, 7, (batch_size,))
    
    next_features, rewards = dynamics_model(state_features, actions)
    
    assert next_features.shape == (batch_size, 64), f"Expected shape ({batch_size}, 64), got {next_features.shape}"
    assert rewards.shape == (batch_size,), f"Expected shape ({batch_size},), got {rewards.shape}"
    print(f"✓ Forward pass works: state {state_features.shape}, actions {actions.shape}")
    print(f"  -> next_features {next_features.shape}, rewards {rewards.shape}")
    
    # Test compute_loss
    next_state_features = torch.randn(batch_size, 64)
    actual_rewards = torch.randn(batch_size)
    
    loss = dynamics_model.compute_loss(state_features, next_state_features, actions, actual_rewards)
    assert loss.item() > 0, "Loss should be positive"
    print(f"✓ Loss computation works: loss = {loss.item():.4f}")
    
    print("✓ DynamicsModel: ALL TESTS PASSED\n")
    return dynamics_model


def test_save_load():
    """Test saving and loading models"""
    print("=" * 60)
    print("TEST 3: Save/Load Models")
    print("=" * 60)
    
    from adapters.transfer_encoder import ObservationEncoder, DynamicsModel, save_transfer_models, load_transfer_models
    
    # Create models
    encoder = ObservationEncoder(input_dim=8, feature_size=64)
    dynamics_model = DynamicsModel(feature_size=64, num_actions=7)
    
    # Save models
    checkpoint_path = "artifacts/test_transfer_models/test_checkpoint.pt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    save_transfer_models(
        encoder, dynamics_model, checkpoint_path,
        input_dim=8, feature_size=64, num_actions=7
    )
    print(f"✓ Models saved to {checkpoint_path}")
    
    # Load models
    loaded_encoder, loaded_dynamics = load_transfer_models(checkpoint_path)
    print(f"✓ Models loaded from {checkpoint_path}")
    
    # Test that loaded models work
    obs = torch.randn(3, 8)
    features_original = encoder(obs)
    features_loaded = loaded_encoder(obs)
    
    assert torch.allclose(features_original, features_loaded, atol=1e-5), "Loaded encoder should match original"
    print(f"✓ Loaded encoder produces same output as original")
    
    state_features = torch.randn(3, 64)
    actions = torch.randint(0, 7, (3,))
    next_orig, rew_orig = dynamics_model(state_features, actions)
    next_loaded, rew_loaded = loaded_dynamics(state_features, actions)
    
    assert torch.allclose(next_orig, next_loaded, atol=1e-5), "Loaded dynamics model should match original"
    assert torch.allclose(rew_orig, rew_loaded, atol=1e-5), "Loaded dynamics model rewards should match"
    print(f"✓ Loaded dynamics model produces same output as original")
    
    # Cleanup
    os.remove(checkpoint_path)
    print(f"✓ Cleanup: removed test checkpoint")
    
    print("✓ Save/Load: ALL TESTS PASSED\n")
    return checkpoint_path


def test_observation_translator_integration():
    """Test integration with ObservationTranslator"""
    print("=" * 60)
    print("TEST 4: ObservationTranslator Integration")
    print("=" * 60)
    
    from adapters.observation_translator import ObservationTranslator
    from adapters.transfer_encoder import ObservationEncoder, DynamicsModel, save_transfer_models
    
    # Create and save models
    encoder = ObservationEncoder(input_dim=8, feature_size=64)
    dynamics_model = DynamicsModel(feature_size=64, num_actions=7)
    
    checkpoint_path = "artifacts/test_transfer_models/test_integration.pt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    save_transfer_models(encoder, dynamics_model, checkpoint_path, input_dim=8, feature_size=64, num_actions=7)
    
    # Test without transfer
    obs_t_no_transfer = ObservationTranslator()
    print(f"✓ ObservationTranslator created (no transfer)")
    
    # Test CBS observation
    cbs_obs = {
        "discovered_node_count": 5,
        "nodes_privilegelevel": np.array([1, 1, 0, 0, 0], dtype=np.int32),
        "discovered_nodes_properties": np.zeros((5, 3), dtype=np.int32),
        "credential_cache_length": 2,
        "_explored_network": type('obj', (object,), {'number_of_edges': lambda: 10})(),
        "probe_result": 0,
        "escalation": 0,
    }
    
    obs_no_transfer = obs_t_no_transfer.from_cbs(cbs_obs)
    assert obs_no_transfer.shape == (8,), f"Expected shape (8,), got {obs_no_transfer.shape}"
    print(f"✓ CBS observation translation (no transfer): {obs_no_transfer.shape}")
    
    # Test with transfer
    obs_t_with_transfer = ObservationTranslator(
        use_transfer=True,
        encoder_path=checkpoint_path
    )
    print(f"✓ ObservationTranslator created (with transfer)")
    
    obs_with_transfer = obs_t_with_transfer.from_cbs(cbs_obs)
    assert obs_with_transfer.shape == (64,), f"Expected shape (64,), got {obs_with_transfer.shape}"
    print(f"✓ CBS observation translation (with transfer): {obs_with_transfer.shape}")
    
    # Test Cyberwheel observation
    cw_obs = np.random.randint(-1, 2, size=(7 * 5 + 1))  # 5 hosts + 1 standalone
    obs_cw_no_transfer = obs_t_no_transfer.from_cw(cw_obs)
    assert obs_cw_no_transfer.shape == (8,), f"Expected shape (8,), got {obs_cw_no_transfer.shape}"
    print(f"✓ Cyberwheel observation translation (no transfer): {obs_cw_no_transfer.shape}")
    
    obs_cw_with_transfer = obs_t_with_transfer.from_cw(cw_obs)
    assert obs_cw_with_transfer.shape == (64,), f"Expected shape (64,), got {obs_cw_with_transfer.shape}"
    print(f"✓ Cyberwheel observation translation (with transfer): {obs_cw_with_transfer.shape}")
    
    # Cleanup
    os.remove(checkpoint_path)
    
    print("✓ ObservationTranslator Integration: ALL TESTS PASSED\n")


def test_transfer_training_utilities():
    """Test transfer training utilities"""
    print("=" * 60)
    print("TEST 5: Transfer Training Utilities")
    print("=" * 60)
    
    from adapters.transfer_training import ReplayBuffer, train_dynamics_model, compute_regularization_loss
    from adapters.transfer_encoder import ObservationEncoder, DynamicsModel
    
    # Test ReplayBuffer
    buffer = ReplayBuffer(capacity=100)
    print(f"✓ ReplayBuffer created")
    
    # Add some transitions
    for i in range(50):
        buffer.push(
            obs=np.random.randn(8),
            action=i % 7,
            next_obs=np.random.randn(8),
            reward=np.random.randn(),
            done=(i % 10 == 0)
        )
    
    assert len(buffer) == 50, f"Expected 50 transitions, got {len(buffer)}"
    print(f"✓ Added 50 transitions to buffer")
    
    # Test sampling
    batch = buffer.sample(10)
    assert len(batch) == 10, f"Expected batch size 10, got {len(batch)}"
    print(f"✓ Sampling works: sampled {len(batch)} transitions")
    
    # Test dynamics model training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ObservationEncoder(input_dim=8, feature_size=64).to(device)
    dynamics_model = DynamicsModel(feature_size=64, num_actions=7).to(device)
    
    losses = train_dynamics_model(
        encoder, dynamics_model, buffer,
        batch_size=10, num_epochs=5, device=device
    )
    
    assert len(losses) == 5, f"Expected 5 losses, got {len(losses)}"
    assert all(l > 0 for l in losses), "All losses should be positive"
    print(f"✓ Dynamics model training works: {len(losses)} epochs, final loss = {losses[-1]:.4f}")
    
    # Test regularization loss
    obs = np.random.randn(8)
    next_obs = np.random.randn(8)
    action = 3
    
    reg_loss = compute_regularization_loss(
        encoder, dynamics_model, obs, next_obs, action, device
    )
    
    assert reg_loss.item() > 0, "Regularization loss should be positive"
    print(f"✓ Regularization loss computation works: loss = {reg_loss.item():.4f}")
    
    print("✓ Transfer Training Utilities: ALL TESTS PASSED\n")


def test_end_to_end():
    """Test end-to-end workflow"""
    print("=" * 60)
    print("TEST 6: End-to-End Workflow")
    print("=" * 60)
    
    from adapters.transfer_encoder import ObservationEncoder, DynamicsModel, save_transfer_models, load_transfer_models
    from adapters.observation_translator import ObservationTranslator
    from adapters.transfer_training import ReplayBuffer, train_dynamics_model, compute_regularization_loss
    
    print("Simulating workflow:")
    print("  1. Create encoder and dynamics model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ObservationEncoder(input_dim=8, feature_size=64).to(device)
    dynamics_model = DynamicsModel(feature_size=64, num_actions=7).to(device)
    print("  ✓ Models created")
    
    print("  2. Collect some transitions (simulated)")
    buffer = ReplayBuffer()
    for i in range(100):
        buffer.push(
            obs=np.random.randn(8),
            action=i % 7,
            next_obs=np.random.randn(8),
            reward=np.random.randn(),
            done=(i % 20 == 0)
        )
    print(f"  ✓ Collected {len(buffer)} transitions")
    
    print("  3. Train dynamics model")
    losses = train_dynamics_model(
        encoder, dynamics_model, buffer,
        batch_size=32, num_epochs=3, device=device
    )
    print(f"  ✓ Dynamics model trained: final loss = {losses[-1]:.4f}")
    
    print("  4. Save models")
    checkpoint_path = "artifacts/test_transfer_models/e2e_test.pt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    save_transfer_models(encoder, dynamics_model, checkpoint_path, input_dim=8, feature_size=64, num_actions=7)
    print(f"  ✓ Models saved")
    
    print("  5. Load models in ObservationTranslator")
    obs_t = ObservationTranslator(use_transfer=True, encoder_path=checkpoint_path)
    print("  ✓ ObservationTranslator with transfer created")
    
    print("  6. Test encoding observations")
    cbs_obs = {
        "discovered_node_count": 3,
        "nodes_privilegelevel": np.array([1, 0, 0], dtype=np.int32),
        "discovered_nodes_properties": np.zeros((3, 3), dtype=np.int32),
        "credential_cache_length": 1,
        "_explored_network": type('obj', (object,), {'number_of_edges': lambda: 5})(),
        "probe_result": 0,
        "escalation": 0,
    }
    encoded_obs = obs_t.from_cbs(cbs_obs)
    assert encoded_obs.shape == (64,), f"Expected (64,), got {encoded_obs.shape}"
    print(f"  ✓ Observation encoded: {cbs_obs['discovered_node_count']} nodes -> {encoded_obs.shape} features")
    
    print("  7. Test regularization loss")
    next_cbs_obs = {
        "discovered_node_count": 4,
        "nodes_privilegelevel": np.array([1, 1, 0, 0], dtype=np.int32),
        "discovered_nodes_properties": np.zeros((4, 3), dtype=np.int32),
        "credential_cache_length": 2,
        "_explored_network": type('obj', (object,), {'number_of_edges': lambda: 8})(),
        "probe_result": 0,
        "escalation": 0,
    }
    # Get raw observations (8-dim) before encoding
    obs_t_raw = ObservationTranslator()  # Without transfer to get raw 8-dim obs
    raw_obs = obs_t_raw.from_cbs(cbs_obs)
    raw_next_obs = obs_t_raw.from_cbs(next_cbs_obs)
    reg_loss = compute_regularization_loss(
        encoder, dynamics_model, raw_obs, raw_next_obs, action=2, device=device
    )
    print(f"  ✓ Regularization loss computed: {reg_loss.item():.4f}")
    
    # Cleanup
    os.remove(checkpoint_path)
    print("  ✓ Cleanup complete")
    
    print("✓ End-to-End Workflow: ALL TESTS PASSED\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("OBSERVATION TRANSFER IMPLEMENTATION - TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        # Test individual components
        encoder = test_encoder()
        dynamics_model = test_dynamics_model()
        test_save_load()
        test_observation_translator_integration()
        test_transfer_training_utilities()
        test_end_to_end()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nThe observation transfer implementation is working correctly.")
        print("You can now use it in your training pipeline.")
        print("\nNext steps:")
        print("  1. Train encoder + dynamics model on Cyberwheel")
        print("  2. Save models using save_transfer_models()")
        print("  3. Load models in ObservationTranslator for CBS evaluation")
        print("  4. Use compute_regularization_loss() during target task training")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

