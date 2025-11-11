import os
import sys
import torch
import pickle
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from pytorch_lightning import seed_everything
from tqdm import tqdm


def generate_candidates_from_model(model_path, args, output_path=None):
    """
    Generate retrieved.pkl from a trained model checkpoint
    
    Args:
        model_path: Path to the .pth model file
        args: Configuration arguments
        output_path: Path to save retrieved.pkl (optional)
    """
    # Import here to avoid argparse conflict
    from config import STATE_DICT_KEY
    from model import LRURec
    from dataloader import dataloader_factory
    from trainer.utils import absolute_recall_mrr_ndcg_for_ks
    
    # Validate model path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading dataset: {args.dataset_code}")
    train_loader, val_loader, test_loader = dataloader_factory(args)
    
    # Create model
    print("Creating model...")
    model = LRURec(args)
    
    # Load model weights
    print("Loading model weights...")
    checkpoint = torch.load(model_path, map_location=device)
    if STATE_DICT_KEY in checkpoint:
        model_state_dict = checkpoint[STATE_DICT_KEY]
    else:
        model_state_dict = checkpoint
    
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Determine output path
    if output_path is None:
        model_dir = os.path.dirname(os.path.dirname(model_path))  # Go up 2 levels from models/best_acc_model.pth
        output_path = os.path.join(model_dir, 'retrieved.pkl')
    
    print(f"\nOutput path: {output_path}")
    
    # Generate candidates
    val_probs, val_labels = [], []
    test_probs, test_labels = [], []
    
    with torch.no_grad():
        print('\n*************** Generating Candidates for Validation Set ***************')
        tqdm_dataloader = tqdm(val_loader)
        val_dataset = val_loader.dataset
        
        for batch_idx, batch in enumerate(tqdm_dataloader):
            # Move batch to device
            seqs, labels = batch
            seqs = seqs.to(device)
            labels = labels.to(device)
            
            # Get scores from model
            scores = model(seqs)[:, -1, :]
            B, L = seqs.shape
            
            # Mask ALL history items for each user
            for i in range(B):
                user_idx = batch_idx * val_loader.batch_size + i
                if user_idx < len(val_dataset.users):
                    user_id = val_dataset.users[user_idx]
                    # Get ALL training items for this user
                    all_history_items = val_dataset.u2seq[user_id]
                    # Mask all history items
                    scores[i, all_history_items] = -1e9
            
            scores[:, 0] = -1e9  # padding
            val_probs.extend(scores.cpu().tolist())
            val_labels.extend(labels.view(-1).cpu().tolist())
        
        # Calculate validation metrics
        val_metrics = absolute_recall_mrr_ndcg_for_ks(
            torch.tensor(val_probs), 
            torch.tensor(val_labels).view(-1), 
            args.metric_ks
        )
        print("\nValidation Metrics:")
        print(val_metrics)

        print('\n****************** Generating Candidates for Test Set ******************')
        tqdm_dataloader = tqdm(test_loader)
        test_dataset = test_loader.dataset
        
        for batch_idx, batch in enumerate(tqdm_dataloader):
            # Move batch to device
            seqs, labels = batch
            seqs = seqs.to(device)
            labels = labels.to(device)
            
            # Get scores from model
            scores = model(seqs)[:, -1, :]
            B, L = seqs.shape
            
            # Mask ALL history items for each user (train + val)
            for i in range(B):
                user_idx = batch_idx * test_loader.batch_size + i
                if user_idx < len(test_dataset.users):
                    user_id = test_dataset.users[user_idx]
                    # Get ALL training + validation items for this user
                    all_history_items = test_dataset.u2seq[user_id] + test_dataset.u2val[user_id]
                    # Mask all history items
                    scores[i, all_history_items] = -1e9
            
            scores[:, 0] = -1e9  # padding
            test_probs.extend(scores.cpu().tolist())
            test_labels.extend(labels.view(-1).cpu().tolist())
        
        # Calculate test metrics
        test_metrics = absolute_recall_mrr_ndcg_for_ks(
            torch.tensor(test_probs), 
            torch.tensor(test_labels).view(-1), 
            args.metric_ks
        )
        print("\nTest Metrics:")
        print(test_metrics)

    # Save to pickle file
    print(f"\nSaving results to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump({
            'val_probs': val_probs,
            'val_labels': val_labels,
            'val_metrics': val_metrics,
            'test_probs': test_probs,
            'test_labels': test_labels,
            'test_metrics': test_metrics
        }, f)
    
    print(f"✓ Successfully saved retrieved.pkl to {output_path}")
    return output_path


if __name__ == "__main__":
    # Manual argument parsing to avoid conflict with config.py
    model_path = None
    dataset = None
    output = None
    device = 'cuda'
    seed = 2020
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--model_path' and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--dataset' and i + 1 < len(sys.argv):
            dataset = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
            output = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--device' and i + 1 < len(sys.argv):
            device = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--seed' and i + 1 < len(sys.argv):
            seed = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] in ['-h', '--help']:
            print("Usage: python generate_retrieved.py --model_path <path> --dataset <name> [options]")
            print("\nRequired arguments:")
            print("  --model_path PATH    Path to the .pth model file")
            print("  --dataset NAME       Dataset name (ml-100k, beauty, yelp2020)")
            print("\nOptional arguments:")
            print("  --output PATH        Output path for retrieved.pkl")
            print("  --device DEVICE      Device to use (cuda or cpu, default: cuda)")
            print("  --seed SEED          Random seed (default: 2020)")
            print("\nExample:")
            print("  python generate_retrieved.py --model_path experiments/lru/beauty/models/best_acc_model.pth --dataset beauty")
            sys.exit(0)
        else:
            i += 1
    
    # Validate required arguments
    if model_path is None:
        print("Error: --model_path is required")
        print("Use --help for usage information")
        sys.exit(1)
    
    if dataset is None:
        print("Error: --dataset is required")
        print("Use --help for usage information")
        sys.exit(1)
    
    if dataset not in ['ml-100k', 'beauty', 'yelp2020']:
        print(f"Error: dataset must be one of: ml-100k, beauty, yelp2020")
        sys.exit(1)
    
    # Clear sys.argv to prevent config.py from parsing it
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]  # Keep only script name
    
    # Now safe to import config
    from config import args as config_args, set_template
    
    # Restore original argv (in case needed later)
    sys.argv = original_argv
    
    # Set up configuration
    config_args.model_code = 'lru'
    config_args.dataset_code = dataset
    config_args.device = device
    config_args.seed = seed
    
    # Set template (batch size, max_len, etc.)
    set_template(config_args)
    
    # Set seed
    seed_everything(config_args.seed)
    
    # Generate candidates
    print("="*80)
    print(f"Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Dataset: {dataset}")
    print(f"  Device: {config_args.device}")
    print(f"  Seed: {config_args.seed}")
    print("="*80)
    
    output_file = generate_candidates_from_model(
        model_path=model_path,
        args=config_args,
        output_path=output
    )
    
    print("\n" + "="*80)
    print("Done! You can now use this file for reranking with LLM.")
    print("="*80)

