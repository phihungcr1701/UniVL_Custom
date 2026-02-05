"""Logging setup for W&B and TensorBoard"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Logger:
    """Unified logger for console, W&B, and TensorBoard"""
    
    def __init__(
        self,
        log_dir: str,
        use_wandb: bool = True,
        use_tensorboard: bool = True,
        wandb_project: str = "univl-msrvtt",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        log_level: str = "INFO",
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup console logging
        self.console_logger = self._setup_console_logger(log_level)
        
        # Setup W&B
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                config=wandb_config,
                dir=str(self.log_dir),
            )
            self.console_logger.info(f"W&B initialized: {wandb.run.name}")
        elif use_wandb and not WANDB_AVAILABLE:
            self.console_logger.warning("W&B requested but not installed. Skipping.")
        
        # Setup TensorBoard
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        if self.use_tensorboard:
            tb_dir = self.log_dir / "tensorboard"
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            self.console_logger.info(f"TensorBoard logging to: {tb_dir}")
        elif use_tensorboard and not TENSORBOARD_AVAILABLE:
            self.console_logger.warning("TensorBoard requested but not installed. Skipping.")
        
        self.step = 0
    
    def _setup_console_logger(self, log_level: str) -> logging.Logger:
        """Setup console logger"""
        logger = logging.getLogger("UniVL")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / "train.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log scalar value to all backends"""
        if step is None:
            step = self.step
        
        if self.use_wandb:
            wandb.log({tag: value}, step=step)
        
        if self.use_tensorboard:
            self.tb_writer.add_scalar(tag, value, step)
    
    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple scalar values"""
        if step is None:
            step = self.step
        
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """Log text to all backends"""
        if step is None:
            step = self.step
        
        if self.use_wandb:
            wandb.log({tag: wandb.Html(text)}, step=step)
        
        if self.use_tensorboard:
            self.tb_writer.add_text(tag, text, step)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration"""
        if self.use_wandb:
            wandb.config.update(config)
        
        self.info("Configuration:")
        for key, value in config.items():
            self.info(f"  {key}: {value}")
    
    def info(self, message: str):
        """Log info message"""
        self.console_logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.console_logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.console_logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.console_logger.debug(message)
    
    def set_step(self, step: int):
        """Set current step"""
        self.step = step
    
    def finish(self):
        """Cleanup logging"""
        if self.use_wandb:
            wandb.finish()
        
        if self.use_tensorboard:
            self.tb_writer.close()


def setup_logger(
    log_dir: str,
    use_wandb: bool = True,
    use_tensorboard: bool = True,
    wandb_project: str = "univl-msrvtt",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None,
    log_level: str = "INFO",
) -> Logger:
    """
    Setup logger with W&B and TensorBoard support
    
    Args:
        log_dir: Directory for logs
        use_wandb: Enable W&B logging
        use_tensorboard: Enable TensorBoard logging
        wandb_project: W&B project name
        wandb_entity: W&B entity/team name
        wandb_run_name: W&B run name
        wandb_config: Configuration dict to log to W&B
        log_level: Logging level (INFO, DEBUG, WARNING, ERROR)
    
    Returns:
        Logger instance
    """
    return Logger(
        log_dir=log_dir,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        wandb_config=wandb_config,
        log_level=log_level,
    )
