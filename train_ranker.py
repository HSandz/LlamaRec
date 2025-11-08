import os
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'  # Disable wandb completely

import argparse
from datasets import DATASETS
from config import *
from model import *
from dataloader import *
from trainer import *

from transformers import BitsAndBytesConfig
from pytorch_lightning import seed_everything
from model import LlamaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    PeftModel,
)


try:
    os.environ['WANDB_PROJECT'] = PROJECT_NAME
except:
    print('WANDB_PROJECT not available, please set it in config.py')


def main(args, export_root=None):
    seed_everything(args.seed)
    if export_root == None:
        export_root = EXPERIMENT_ROOT + '/' + args.llm_base_model.split('/')[-1] + '/' + args.dataset_code

    # Validate checkpoint path if provided
    checkpoint_path = None
    if args.resume_from_checkpoint:
        if os.path.isdir(args.resume_from_checkpoint):
            # This is a Hugging Face checkpoint directory
            checkpoint_path = args.resume_from_checkpoint
            print(f"✓ Will resume training from checkpoint: {checkpoint_path}")
        else:
            print(f"⚠ Warning: Checkpoint path not found or not a directory: {args.resume_from_checkpoint}")
            print("  Starting training from scratch instead.")
            checkpoint_path = None

    train_loader, val_loader, test_loader, tokenizer, test_retrieval = dataloader_factory(args)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load base model
    model = LlamaForCausalLM.from_pretrained(
        args.llm_base_model,
        quantization_config=bnb_config,
        device_map='auto',
        cache_dir=args.llm_cache_dir,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # Check if we should load from a checkpoint with LoRA weights
    if checkpoint_path and os.path.exists(os.path.join(checkpoint_path, 'adapter_model.bin')):
        # Load model with existing LoRA adapters from checkpoint
        print(f"✓ Loading LoRA adapters from checkpoint: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
        print("✓ LoRA adapters loaded successfully")
    else:
        # Initialize fresh LoRA adapters
        print("✓ Initializing new LoRA adapters")
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
    
    model.print_trainable_parameters()
    model.config.use_cache = False
    
    trainer = LLMTrainer(args, model, train_loader, val_loader, test_loader, tokenizer, export_root, args.use_wandb)
    
    # Pass checkpoint_path to train() method for proper resume
    trainer.train(resume_from_checkpoint=checkpoint_path)
    trainer.test(test_retrieval)


if __name__ == "__main__":
    args.model_code = 'llm'
    set_template(args)
    main(args, export_root=None)
