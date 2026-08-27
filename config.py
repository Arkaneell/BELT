import os
import torch

# Constants from CELL 1
NUM_CLASSES = 2
NUM_MEMBERS = 5
IMAGE_DIM = 1280
TEXT_DIM = 312
HIDDEN_DIM = 256
EMBED_DIM = 64
NUM_HEADS = 4
DROPOUT = 0.1
BATCH_SIZE = 32
MEMBER_EPOCHS = 100
ENSEMBLE_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_TEXT_LENGTH = 64
SEED = 42

# Local Path Configuration (Overridable via CLI in main.py)
# Defaulting to a 'local_data' folder in current working dir for local execution
BASE = os.getenv('BELT_BASE', 'local_data')
IMG_ROOT = BASE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'

def ensure_dirs():
    for folder in ['data/embeddings', 'checkpoints', 'plots', 'results']:
        os.makedirs(os.path.join(BASE, folder), exist_ok=True)
