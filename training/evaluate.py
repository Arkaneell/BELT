import torch
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix, roc_auc_score
)
from models.coral import decode_coral
from config import DEVICE, USE_AMP

@torch.no_grad()
def evaluate_member(member, loader):
    member.eval()
    predictions, targets = [], []
    for image, text, mask, label, event in loader:
        image, text, mask = image.to(DEVICE), text.to(DEVICE), mask.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits = member(image, text, mask)
        predictions.extend(decode_coral(logits).cpu().numpy())
        targets.extend(label.numpy())
    return f1_score(targets, predictions, average='macro', zero_division=0)

@torch.no_grad()
def collect_logits(model, loader):
    model.eval()
    all_logits, all_targets = [], []
    for image, text, mask, label, event in loader:
        image, text, mask = image.to(DEVICE), text.to(DEVICE), mask.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits = model(image, text, mask)
        all_logits.append(logits.float().cpu())
        all_targets.append(label.cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)

def evaluate_model(model, loader, threshold=0.5):
    logits, targets = collect_logits(model, loader)
    preds = decode_coral(logits, threshold=threshold)
    targets_np, preds_np = targets.numpy(), preds.numpy()
    return {
        'macro_f1': f1_score(targets_np, preds_np, average='macro', zero_division=0),
        'weighted_f1': f1_score(targets_np, preds_np, average='weighted', zero_division=0),
        'accuracy': accuracy_score(targets_np, preds_np),
        'mcc': matthews_corrcoef(targets_np, preds_np),
        'kappa': cohen_kappa_score(targets_np, preds_np),
        'auc': roc_auc_score(targets_np, torch.sigmoid(logits[:, 0]).numpy()),
        'targets': targets_np, 'preds': preds_np,
    }

def optimize_coral_threshold(logits, targets):
    best_result = None
    for t in np.arange(0.20, 0.81, 0.01):
        preds = decode_coral(logits, threshold=t).numpy()
        macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
        acc = accuracy_score(targets, preds)
        score = macro_f1 + 0.20 * acc
        if best_result is None or score > best_result['score']:
            best_result = {'t': float(t), 'macro_f1': float(macro_f1), 'accuracy': float(acc), 'score': float(score)}
    return best_result['t']
