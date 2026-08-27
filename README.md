# BELT Inform Modularized Project

Multimodal Bidirectional Ensemble Learning Transformer for informativeness classification.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare Data:
   Ensure your data is structured as:
   - `data_dir/processed/inform_train.csv`
   - `data_dir/processed/inform_dev.csv`
   - `data_dir/processed/inform_test.csv`
   - Images located in the `base` directory (same as `IMG_ROOT`).

## Usage
Run the full pipeline from the CLI:
```bash
python main.py --data-dir "C:/path/to/your/data" --base "C:/path/to/output"
```

## Structure
- `config.py`: Model and training hyperparameters.
- `utils.py`: Text cleaning and utility functions.
- `data/`: Feature extraction and PyTorch dataset implementation.
- `models/`: BELT, CORAL, and Ensemble architectures.
- `training/`: Training loops and evaluation metrics.
- `pipeline.py`: High-level orchestration of the BELT process.
