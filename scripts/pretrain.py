"""Pretrain UniVL on MSRVTT with MLM and MFM"""

import argparse
import random
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import UniVLModel
from src.data import create_dataloader
from src.training import Trainer, Evaluator
from src.utils import load_config, setup_logger


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Pretrain UniVL on MSRVTT")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_msrvtt.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(config.seed)
    
    # Setup device
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        n_gpu = 1
    
    config.training.local_rank = args.local_rank
    config.training.n_gpu = n_gpu
    
    # Setup logger
    logger = setup_logger(
        log_dir=config.training.output_dir,
        use_wandb=config.logging.use_wandb,
        use_tensorboard=config.logging.use_tensorboard,
        wandb_project=config.logging.wandb_project,
        wandb_entity=config.logging.wandb_entity,
        wandb_run_name=config.logging.wandb_run_name or "pretrain",
        wandb_config=config.to_dict(),
        log_level=config.logging.log_level,
    )
    
    logger.info("=" * 80)
    logger.info("Pretraining UniVL on MSRVTT")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Number of GPUs: {n_gpu}")
    logger.info(f"Output directory: {config.training.output_dir}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader = create_dataloader(
        config=config,
        tokenizer=tokenizer,
        split="train",
        is_pretrain=True,
    )
    
    val_loader = create_dataloader(
        config=config,
        tokenizer=tokenizer,
        split="val",
        is_pretrain=True,
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = UniVLModel(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Load checkpoint if resuming
    if args.resume_from:
        logger.info(f"Loading checkpoint from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger,
        device=device,
    )
    
    # Create evaluator (optional for pretrain, mainly for monitoring)
    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        config=config,
        logger=logger,
        device=device,
    )
    
    # Start training
    logger.info("Starting pretraining...")
    trainer.train(evaluator=evaluator)
    
    logger.info("Pretraining completed!")
    logger.finish()


if __name__ == "__main__":
    main()
