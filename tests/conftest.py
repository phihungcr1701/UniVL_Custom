"""Unit tests setup file"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Set test environment variables
os.environ["WANDB_MODE"] = "disabled"  # Disable W&B in tests
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
