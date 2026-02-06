"""Evaluator for UniVL model"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    COCO_EVAL_AVAILABLE = True
except ImportError:
    COCO_EVAL_AVAILABLE = False
    print("Warning: pycocoevalcap not installed. Metrics will not be computed.")

from ..utils.logger import Logger


class Evaluator:
    """Evaluator for caption generation"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config,
        logger: Logger,
        device: str = "cuda",
    ):
        """
        Args:
            model: UniVL model
            tokenizer: HuggingFace tokenizer
            config: UniVLConfig object
            logger: Logger instance
            device: Device to evaluate on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
        self.device = device
        
        self.eval_config = config.evaluation
        self.output_dir = Path(config.training.output_dir)
        
        # Evaluation metrics
        if COCO_EVAL_AVAILABLE:
            self.scorers = {
                'bleu': Bleu(4),
                'meteor': Meteor(),
                'rouge': Rouge(),
                'cider': Cider(),
            }
        else:
            self.scorers = {}
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        global_step: int,
        save_predictions: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate model on validation/test set
        
        Args:
            dataloader: DataLoader for evaluation
            global_step: Current training step
            save_predictions: Whether to save predictions to file
        
        Returns:
            Dictionary of evaluation metrics
            
        Note:
            Currently only uses single reference per video from batch.
            For accurate MSRVTT evaluation, should use all ~20 references per video.
        """
        self.model.eval()
        
        # Unwrap DataParallel to access custom methods
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        all_predictions = []
        all_references = []
        
        self.logger.info("Generating captions...")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move batch to device
            video = batch['video'].to(self.device)
            video_mask = batch['video_mask'].to(self.device)
            
            # Generate captions (use unwrapped model)
            generated_ids = model.generate_caption(
                video=video,
                video_mask=video_mask,
                max_length=self.eval_config.max_gen_length,
                beam_size=self.eval_config.beam_size,
                bos_token_id=self.tokenizer.cls_token_id,
                eos_token_id=self.tokenizer.sep_token_id,
                tokenizer=self.tokenizer,  # Pass tokenizer for beam search
            )
            
            # Decode predictions
            for i in range(len(generated_ids)):
                # Decode generated caption
                pred_tokens = generated_ids[i].cpu().tolist()
                pred_text = self.tokenizer.decode(
                    pred_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                
                # Get reference caption(s)
                # Note: For MSRVTT, we need multiple references per video
                # This is a simplified version - in practice, need to get all refs
                ref_tokens = batch['output_caption_ids'][i].cpu().tolist()
                ref_text = self.tokenizer.decode(
                    ref_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                
                all_predictions.append(pred_text)
                all_references.append([ref_text])  # List of references
        
        # Save predictions
        if save_predictions:
            self._save_predictions(all_predictions, all_references, global_step)
        
        # Compute metrics
        metrics = self._compute_metrics(all_predictions, all_references)
        
        # Log metrics
        self.logger.info("Evaluation results:")
        for metric_name, score in metrics.items():
            self.logger.info(f"  {metric_name}: {score:.4f}")
            self.logger.log_scalar(f"eval/{metric_name}", score, global_step)
        
        # Log sample predictions
        self._log_samples(all_predictions, all_references, num_samples=5)
        
        return metrics
    
    def _compute_metrics(
        self,
        predictions: List[str],
        references: List[List[str]],
    ) -> Dict[str, float]:
        """Compute caption evaluation metrics"""
        if not COCO_EVAL_AVAILABLE or not self.scorers:
            self.logger.warning("Evaluation metrics not available")
            return {}
        
        # Format for pycocoevalcap
        # gts: {id: [ref1, ref2, ...]}
        # res: {id: [prediction]}
        gts = {i: refs for i, refs in enumerate(references)}
        res = {i: [pred] for i, pred in enumerate(predictions)}
        
        metrics = {}
        
        # Compute each metric
        for metric_name, scorer in self.scorers.items():
            try:
                score, scores = scorer.compute_score(gts, res)
                if isinstance(score, list):
                    # BLEU returns list of scores for n-grams
                    for i, s in enumerate(score):
                        metrics[f'bleu{i+1}'] = s
                else:
                    metrics[metric_name] = score
            except Exception as e:
                self.logger.warning(f"Failed to compute {metric_name}: {e}")
        
        return metrics
    
    def _save_predictions(
        self,
        predictions: List[str],
        references: List[List[str]],
        global_step: int,
    ):
        """Save predictions and references to files"""
        output_dir = self.output_dir / f"predictions_step_{global_step}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        pred_file = output_dir / "predictions.txt"
        with open(pred_file, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(pred + '\n')
        
        # Save references
        ref_file = output_dir / "references.txt"
        with open(ref_file, 'w', encoding='utf-8') as f:
            for refs in references:
                f.write(' ||| '.join(refs) + '\n')
        
        # Save as JSON for easier parsing
        results = {
            'step': global_step,
            'predictions': predictions,
            'references': references,
        }
        json_file = output_dir / "results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Predictions saved to {output_dir}")
    
    def _log_samples(
        self,
        predictions: List[str],
        references: List[List[str]],
        num_samples: int = 5,
    ):
        """Log sample predictions to console"""
        self.logger.info("\nSample predictions:")
        for i in range(min(num_samples, len(predictions))):
            self.logger.info(f"\nSample {i+1}:")
            self.logger.info(f"  Prediction: {predictions[i]}")
            self.logger.info(f"  Reference:  {references[i][0]}")
