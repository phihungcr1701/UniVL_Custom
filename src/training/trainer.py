"""Trainer for UniVL model"""

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from ..utils.logger import Logger
from ..utils.checkpoint import save_checkpoint


class Trainer:
    """Trainer for UniVL model (pretrain and fine-tuning)"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config,
        logger: Logger,
        device: str = "cuda",
    ):
        """
        Args:
            model: UniVL model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: UniVLConfig object
            logger: Logger instance
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.device = device
        
        self.training_config = config.training
        self.model_config = config.model
        
        # Move model to device
        self.model.to(device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision training
        self.use_amp = self.training_config.use_amp
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = 0.0
    
    def _setup_optimizer(self):
        """Setup optimizer with different learning rates for BERT vs other modules"""
        # Separate parameters into BERT and non-BERT
        bert_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'text_encoder.bert' in name:
                bert_params.append(param)
            else:
                other_params.append(param)
        
        # Create optimizer with different learning rates
        coef_lr = self.training_config.coef_lr
        optimizer = torch.optim.AdamW([
            {'params': bert_params, 'lr': self.training_config.learning_rate * coef_lr},
            {'params': other_params, 'lr': self.training_config.learning_rate},
        ], weight_decay=self.training_config.weight_decay, eps=self.training_config.adam_epsilon)
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup"""
        num_training_steps = len(self.train_loader) * self.training_config.epochs
        num_warmup_steps = int(num_training_steps * self.training_config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        return scheduler
    
    def train_epoch(self, epoch: int):
        """Train one epoch"""
        self.model.train()
        self.current_epoch = epoch
        
        total_loss = 0.0
        caption_loss_sum = 0.0
        mlm_loss_sum = 0.0
        mfm_loss_sum = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.training_config.epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs['loss']
            else:
                outputs = self.model(**batch)
                loss = outputs['loss']
            
            # Gradient accumulation
            loss = loss / self.training_config.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Accumulate losses
            total_loss += loss.item() * self.training_config.gradient_accumulation_steps
            caption_loss_sum += outputs.get('caption_loss', 0).item()
            if 'mlm_loss' in outputs:
                mlm_loss_sum += outputs['mlm_loss'].item()
            if 'mfm_loss' in outputs:
                mfm_loss_sum += outputs['mfm_loss'].item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
            })
            
            # Logging
            if self.global_step % self.training_config.logging_steps == 0:
                metrics = {
                    'train/loss': loss.item() * self.training_config.gradient_accumulation_steps,
                    'train/caption_loss': outputs.get('caption_loss', 0).item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                }
                if 'mlm_loss' in outputs:
                    metrics['train/mlm_loss'] = outputs['mlm_loss'].item()
                if 'mfm_loss' in outputs:
                    metrics['train/mfm_loss'] = outputs['mfm_loss'].item()
                
                self.logger.log_scalars(metrics, self.global_step)
        
        # Epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_caption_loss = caption_loss_sum / len(self.train_loader)
        
        epoch_metrics = {
            'epoch': epoch + 1,
            'train/epoch_loss': avg_loss,
            'train/epoch_caption_loss': avg_caption_loss,
        }
        
        if self.model_config.pretrain_mode:
            epoch_metrics['train/epoch_mlm_loss'] = mlm_loss_sum / len(self.train_loader)
            epoch_metrics['train/epoch_mfm_loss'] = mfm_loss_sum / len(self.train_loader)
        
        self.logger.log_scalars(epoch_metrics, self.global_step)
        self.logger.info(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint"""
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            step=self.global_step,
            metrics=metrics,
            save_dir=self.training_config.output_dir,
            filename=f"checkpoint_epoch_{epoch+1}.pth",
            is_best=is_best,
        )
    
    def train(self, evaluator=None):
        """Full training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {self.training_config.epochs}")
        self.logger.info(f"Total steps: {len(self.train_loader) * self.training_config.epochs}")
        
        for epoch in range(self.training_config.epochs):
            # Train epoch
            train_loss = self.train_epoch(epoch)
            
            # Evaluate
            if evaluator is not None and self.val_loader is not None:
                self.logger.info("Running evaluation...")
                eval_metrics = evaluator.evaluate(self.val_loader, self.global_step)
                
                # Check if best model
                current_metric = eval_metrics.get('bleu4', 0.0)
                is_best = current_metric > self.best_metric
                if is_best:
                    self.best_metric = current_metric
                    self.logger.info(f"New best model! BLEU-4: {current_metric:.4f}")
                
                # Save checkpoint
                self.save_checkpoint(
                    epoch=epoch,
                    metrics=eval_metrics,
                    is_best=is_best,
                )
            else:
                # Save checkpoint without evaluation
                self.save_checkpoint(
                    epoch=epoch,
                    metrics={'train_loss': train_loss},
                    is_best=False,
                )
        
        self.logger.info("Training completed!")
        if self.best_metric > 0:
            self.logger.info(f"Best BLEU-4: {self.best_metric:.4f}")
