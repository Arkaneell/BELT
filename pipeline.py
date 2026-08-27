import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

from config import (
    BASE, BATCH_SIZE, NUM_MEMBERS, DEVICE, 
    NUM_CLASSES, SEED
)
from utils import set_seed, clean_dataframe
from data.features import extract_and_cache
from data.datasets import InformDataset, get_class_weights, balanced_bootstrap_sample
from training.train import train_member, train_ensemble
from training.evaluate import evaluate_model, optimize_coral_threshold
from sklearn.metrics import classification_report, confusion_matrix

def run_pipeline(data_dir):
    # 1. Setup
    set_seed(SEED)
    from config import ensure_dirs
    ensure_dirs()

    # 2. Data Loading & Cleaning
    df_train = pd.read_csv(os.path.join(data_dir, 'processed/inform_train.csv'))
    df_dev = pd.read_csv(os.path.join(data_dir, 'processed/inform_dev.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'processed/inform_test.csv'))

    df_train = clean_dataframe(df_train)
    df_dev = clean_dataframe(df_dev)
    df_test = clean_dataframe(df_test)

    # 3. Feature Extraction & Caching
    train_data = extract_and_cache(df_train, os.path.join(BASE, 'data/embeddings/inform_train_belt.pkl'))
    dev_data = extract_and_cache(df_dev, os.path.join(BASE, 'data/embeddings/inform_dev_belt.pkl'))
    test_data = extract_and_cache(df_test, os.path.join(BASE, 'data/embeddings/inform_test_belt.pkl'))

    # 4. Dataset & Loaders
    events = sorted(list(set(train_data['events'] + dev_data['events'] + test_data['events'])))
    event2idx = {e: i for i, e in enumerate(events)}

    train_dataset = InformDataset(train_data, event2idx)
    dev_dataset = InformDataset(dev_data, event2idx)
    test_dataset = InformDataset(test_data, event2idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    class_weights = get_class_weights(train_data['labels'])

    # 5. Member Training
    class_pools = {c: np.where(train_data['labels'] == c)[0] for c in range(NUM_CLASSES)}
    samples_per_class = min(len(v) for v in class_pools.values())
    
    member_indices = [balanced_bootstrap_sample(class_pools, samples_per_class, seed=4000 + i) for i in range(NUM_MEMBERS)]
    member_loaders = [DataLoader(Subset(train_dataset, idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True) for idx in member_indices]

    members = []
    for i in range(NUM_MEMBERS):
        print(f"\nTRAINING MEMBER {i+1}/{NUM_MEMBERS}")
        members.append(train_member(i, member_loaders[i], dev_loader, class_weights))

    # 6. Ensemble Training
    print("\nTRAINING EMBEDDING ENSEMBLE")
    ensemble, best_dev_f1 = train_ensemble(members, train_loader, dev_loader, class_weights)

    # 7. Threshold Optimization
    from training.evaluate import collect_logits
    dev_logits, dev_targets = collect_logits(ensemble, dev_loader)
    best_t = optimize_coral_threshold(dev_logits, dev_targets.numpy())
    print(f"Best DEV threshold: t={best_t:.3f}")

    # 8. Final Test Evaluation
    test_result = evaluate_model(ensemble, test_loader, threshold=best_t)
    
    print("\nFINAL TEST RESULTS")
    print(f"Macro F1 : {test_result['macro_f1']:.4f}")
    print(classification_report(test_result['targets'], test_result['preds'], target_names=["Not Informative", "Informative"], digits=4, zero_division=0))
    print(confusion_matrix(test_result['targets'], test_result['preds']))

    # 9. Saving
    results_payload = {
        'macro_f1': float(test_result['macro_f1']), 
        'weighted_f1': float(test_result['weighted_f1']),
        'accuracy': float(test_result['accuracy']), 
        'mcc': float(test_result['mcc']),
        'kappa': float(test_result['kappa']), 
        'auc': float(test_result['auc']),
        'threshold': best_t, 
        'ensemble_weights': ensemble.get_weights().detach().cpu().numpy().tolist(),
        'num_members': NUM_MEMBERS, 
        'embedding_dimension': 64, 
        'bootstrap_size_per_class': samples_per_class
    }
    with open(os.path.join(BASE, 'results/belt_inform_results.json'), 'w') as f:
        json.dump(results_payload, f, indent=2)
    
    torch.save(ensemble.state_dict(), os.path.join(BASE, 'checkpoints/belt_inform.pth'))
    print("\nSaved results and model.")
