import argparse
import os
from pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="BELT Inform Multimodal Classifier Pipeline")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to directory containing processed/inform_*.csv")
    parser.add_argument("--base", type=str, default="local_data", help="Base output directory for embeddings, checkpoints, and results")
    
    args = parser.parse_args()
    
    # Override config.BASE for the session
    import config
    config.BASE = args.base
    config.IMG_ROOT = args.base # matching notebook's BASE = IMG_ROOT
    
    run_pipeline(args.data_dir)

if __name__ == "__main__":
    main()
