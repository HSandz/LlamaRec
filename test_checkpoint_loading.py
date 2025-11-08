"""
Test script to validate checkpoint loading logic for LLM finetuning
Run this before training to ensure checkpoint resume works correctly
"""

import os
import sys
import torch
from pathlib import Path


def test_checkpoint_structure(checkpoint_path):
    """
    Test if checkpoint directory has correct structure
    """
    print(f"\n{'='*60}")
    print(f"Testing checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(checkpoint_path):
        print("❌ FAIL: Checkpoint path does not exist")
        return False
    
    if not os.path.isdir(checkpoint_path):
        print("❌ FAIL: Checkpoint path is not a directory")
        return False
    
    print("✅ PASS: Checkpoint directory exists")
    
    # Required files for HuggingFace checkpoint
    required_files = {
        'adapter_model.bin': 'LoRA adapter weights',
        'adapter_config.json': 'LoRA configuration',
        'trainer_state.json': 'Training state',
    }
    
    optional_files = {
        'optimizer.pt': 'Optimizer state',
        'scheduler.pt': 'LR scheduler state',
        'rng_state.pth': 'Random number generator state',
        'training_args.bin': 'Training arguments',
    }
    
    all_pass = True
    
    print("\n--- Required Files ---")
    for filename, description in required_files.items():
        filepath = os.path.join(checkpoint_path, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✅ {filename:30s} ({description}) - {size:,} bytes")
        else:
            print(f"❌ {filename:30s} ({description}) - MISSING")
            all_pass = False
    
    print("\n--- Optional Files ---")
    for filename, description in optional_files.items():
        filepath = os.path.join(checkpoint_path, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✅ {filename:30s} ({description}) - {size:,} bytes")
        else:
            print(f"⚠  {filename:30s} ({description}) - Not found (training state may not resume completely)")
    
    return all_pass


def test_adapter_loadable(checkpoint_path):
    """
    Test if adapter_model.bin is loadable
    """
    print(f"\n--- Testing Adapter Loading ---")
    adapter_path = os.path.join(checkpoint_path, 'adapter_model.bin')
    
    if not os.path.exists(adapter_path):
        print("⚠  Skipping: adapter_model.bin not found")
        return None
    
    try:
        state_dict = torch.load(adapter_path, map_location='cpu')
        print(f"✅ Adapter loaded successfully")
        print(f"   Number of parameters: {len(state_dict)}")
        
        # Check if it's LoRA weights (should have 'lora' in keys)
        lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
        if lora_keys:
            print(f"   LoRA parameters found: {len(lora_keys)}")
            print(f"   Sample keys: {lora_keys[:3]}")
        else:
            print("⚠  Warning: No 'lora' keys found - might not be LoRA weights")
        
        return True
    except Exception as e:
        print(f"❌ FAIL: Cannot load adapter_model.bin: {e}")
        return False


def test_config_readable(checkpoint_path):
    """
    Test if adapter_config.json is readable
    """
    print(f"\n--- Testing Configuration ---")
    config_path = os.path.join(checkpoint_path, 'adapter_config.json')
    
    if not os.path.exists(config_path):
        print("⚠  Skipping: adapter_config.json not found")
        return None
    
    try:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Configuration loaded successfully")
        
        # Print important LoRA config
        important_keys = ['r', 'lora_alpha', 'lora_dropout', 'target_modules', 'bias', 'task_type']
        for key in important_keys:
            if key in config:
                print(f"   {key:20s}: {config[key]}")
        
        return True
    except Exception as e:
        print(f"❌ FAIL: Cannot read adapter_config.json: {e}")
        return False


def test_training_state(checkpoint_path):
    """
    Test if trainer_state.json contains valid training state
    """
    print(f"\n--- Testing Training State ---")
    state_path = os.path.join(checkpoint_path, 'trainer_state.json')
    
    if not os.path.exists(state_path):
        print("⚠  Skipping: trainer_state.json not found")
        return None
    
    try:
        import json
        with open(state_path, 'r') as f:
            state = json.load(f)
        
        print(f"✅ Training state loaded successfully")
        
        # Print important training info
        if 'epoch' in state:
            print(f"   Epoch: {state['epoch']}")
        if 'global_step' in state:
            print(f"   Global step: {state['global_step']}")
        if 'best_metric' in state:
            print(f"   Best metric: {state.get('best_metric')}")
        if 'best_model_checkpoint' in state:
            print(f"   Best model checkpoint: {state['best_model_checkpoint']}")
        
        # Check log history
        if 'log_history' in state and state['log_history']:
            print(f"   Number of logged steps: {len(state['log_history'])}")
            last_log = state['log_history'][-1]
            print(f"   Last logged step: {last_log.get('step', 'N/A')}")
            if 'loss' in last_log:
                print(f"   Last training loss: {last_log['loss']:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ FAIL: Cannot read trainer_state.json: {e}")
        return False


def main():
    """
    Main test function
    """
    print("\n" + "="*60)
    print("LLM Checkpoint Validation Tool")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\nUsage: python test_checkpoint_loading.py <checkpoint_path>")
        print("\nExample:")
        print("  python test_checkpoint_loading.py experiments/Llama-2-7b-hf/beauty/checkpoint-100")
        
        # Try to find checkpoints automatically
        print("\n--- Searching for checkpoints ---")
        experiment_root = Path('experiments')
        if experiment_root.exists():
            checkpoints = list(experiment_root.rglob('checkpoint-*'))
            if checkpoints:
                print(f"Found {len(checkpoints)} checkpoint(s):")
                for cp in checkpoints[:10]:  # Show max 10
                    print(f"  - {cp}")
                if len(checkpoints) > 10:
                    print(f"  ... and {len(checkpoints) - 10} more")
            else:
                print("No checkpoints found in experiments/")
        else:
            print("experiments/ directory not found")
        
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    # Run all tests
    results = {}
    results['structure'] = test_checkpoint_structure(checkpoint_path)
    results['adapter'] = test_adapter_loadable(checkpoint_path)
    results['config'] = test_config_readable(checkpoint_path)
    results['state'] = test_training_state(checkpoint_path)
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Skipped: {skipped}")
    
    if failed == 0 and results['structure']:
        print("\n✅ Checkpoint is valid and ready for resume!")
        print(f"\nTo resume training, use:")
        print(f"  python train_ranker.py --resume_from_checkpoint {checkpoint_path}")
    elif results['structure'] is False:
        print("\n❌ Checkpoint has structural issues - cannot resume training")
    else:
        print("\n⚠  Checkpoint may have issues - review warnings above")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
