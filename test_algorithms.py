#!/usr/bin/env python3
"""
Test script to verify that all algorithms work correctly
"""
import torch
from src.model import FTConfig, FeatureTokenizerTransformer
from src.algorithms import BaseAlgorithm, ErmAlgorithm, RExAlgorithm, IBIRMAlgorithm

def test_algorithm_creation():
    """Test that all algorithms can be created correctly"""
    print("Testing algorithm creation...")
    
    # Test ERM
    config_erm = FTConfig(
        n_cont_features=1,
        cat_cardinalities=[2],
        n_blocks=2,
        lr=0.001,
        weight_decay=1e-5,
        batch_size=32,
        d_out=1,
        algorithm_type="erm"
    )
    model_erm = FeatureTokenizerTransformer(config_erm)
    print(f"✓ ERM algorithm: {type(model_erm.algorithm).__name__}")
    
    # Test REX
    config_rex = FTConfig(
        n_cont_features=1,
        cat_cardinalities=[2], 
        n_blocks=2,
        lr=0.001,
        weight_decay=1e-5,
        batch_size=32,
        d_out=1,
        algorithm_type="rex",
        rex_weight=1.0,
        rex_penalty_anneal_iters=100
    )
    model_rex = FeatureTokenizerTransformer(config_rex)
    print(f"✓ REX algorithm: {type(model_rex.algorithm).__name__}")
    
    # Test IB_IRM
    config_ib_irm = FTConfig(
        n_cont_features=1,
        cat_cardinalities=[2],
        n_blocks=2,
        lr=0.001,
        weight_decay=1e-5,
        batch_size=32,
        d_out=1,
        algorithm_type="ib_irm",
        irm_weight=1.0,
        ib_weight=0.1,
        irm_penalty_anneal_iters=100,
        ib_penalty_anneal_iters=100
    )
    model_ib_irm = FeatureTokenizerTransformer(config_ib_irm)
    print(f"✓ IB_IRM algorithm: {type(model_ib_irm.algorithm).__name__}")

def test_algorithm_interfaces():
    """Test that all algorithms implement the required interface"""
    print("\nTesting algorithm interfaces...")
    
    # Create dummy data
    batch_size = 8
    X_cont = torch.randn(batch_size, 1)
    X_cat = torch.randint(0, 2, (batch_size, 1))
    y = torch.randn(batch_size, 1)
    e = torch.randint(0, 2, (batch_size,))  # Environment IDs
    
    loss_fn = torch.nn.MSELoss()
    
    # Test each algorithm
    algorithms = {
        "ERM": ErmAlgorithm(),
        "REX": RExAlgorithm(rex_weight=1.0, rex_penalty_anneal_iters=100),
        "IB_IRM": IBIRMAlgorithm(irm_weight=1.0, ib_weight=0.1)
    }
    
    for name, alg in algorithms.items():
        print(f"Testing {name}...")
        
        # Test basic interface methods
        assert hasattr(alg, 'compute_loss'), f"{name} missing compute_loss"
        assert hasattr(alg, 'compute_penalty'), f"{name} missing compute_penalty"
        assert hasattr(alg, 'is_enabled'), f"{name} missing is_enabled"
        assert hasattr(alg, 'should_reset_optimizer'), f"{name} missing should_reset_optimizer"
        
        # Test compute_loss and compute_penalty
        with torch.no_grad():
            predictions = torch.randn(batch_size, 1)
            features = torch.randn(batch_size, 4) if name == "IB_IRM" else None
            
            # Test compute_loss
            loss = alg.compute_loss(
                predictions, y, e, loss_fn, 
                update_count=0, features=features
            )
            assert isinstance(loss, torch.Tensor), f"{name} compute_loss should return tensor"
            
            # Test compute_penalty
            penalty = alg.compute_penalty(
                predictions, y, e, loss_fn, features=features
            )
            if name == "IB_IRM":
                assert isinstance(penalty, tuple) and len(penalty) == 2, f"{name} should return tuple of penalties"
            else:
                assert isinstance(penalty, torch.Tensor), f"{name} should return penalty tensor"
            
            # Test other methods
            assert isinstance(alg.is_enabled(), bool), f"{name} is_enabled should return bool"
            assert isinstance(alg.should_reset_optimizer(0), bool), f"{name} should_reset_optimizer should return bool"
        
        print(f"✓ {name} interface test passed")

if __name__ == "__main__":
    print("=" * 50)
    print("ALGORITHM STRUCTURE TEST")
    print("=" * 50)
    
    try:
        test_algorithm_creation()
        test_algorithm_interfaces()
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("Algorithm structure is working correctly.")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()