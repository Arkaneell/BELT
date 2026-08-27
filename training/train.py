import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from config import (
    DEVICE, USE_AMP, MEMBER_EPOCHS, ENSEMBLE_EPOCHS, 
    LEARNING_RATE, WEIGHT_DECAY, SEED
)
from models.belt import BELT
from models.ensemble import EmbeddingEnsemble
from models.coral import WeightedBCEWithLogitsLoss
from training.evaluate import evaluate_member, evaluate_model
from utils import set_seed

def train_member(member_id, loader, dev_loader, class_weights, epochs=MEMBER_EPOCHS, patience_limit=50):
    set_seed(SEED + member_id)
    member = BELT().to(DEVICE)
    criterion = WeightedBCEWithLogitsLoss(pos_weight=class_weights[1])
    optimizer = AdamW(member.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    
    best_f1, best_state, patience = -np.inf, None, 0
    for epoch in range(epochs):
        member.train()
        running_loss, samples = 0.0, 0
        for image, text, mask, label, event in loader:
            image, text, mask, label = image.to(DEVICE), text.to(DEVICE), mask.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                logits = member(image, text, mask)
                loss = criterion(logits, label)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(member.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * label.size(0)
            samples += label.size(0)
        
        dev_f1 = evaluate_member(member, dev_loader)
        print(f"Member {member_id+1} | Epoch {epoch+1:02d} | Loss {running_loss/samples:.4f} | DEV Macro-F1 {dev_f1:.4f}")
        if dev_f1 > best_f1:
            best_f1, patience = dev_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in member.state_dict().items()}
        else:
            patience += 1
        if patience >= patience_limit:
            print("Early stopping."); break
            
    member.load_state_dict(best_state)
    for p in member.parameters(): p.requires_grad = False
    member.eval()
    print(f"Member {member_id+1} best DEV Macro-F1: {best_f1:.4f}")
    return member

def train_ensemble(members, train_loader, dev_loader, class_weights, epochs=ENSEMBLE_EPOCHS, patience_limit=50):
    ensemble = EmbeddingEnsemble(members).to(DEVICE)
    criterion = WeightedBCEWithLogitsLoss(pos_weight=class_weights[1])
    optimizer = AdamW([ensemble.weight_logits, *ensemble.head.parameters()], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    
    best_f1, best_state, patience = -np.inf, None, 0
    for epoch in range(epochs):
        ensemble.train()
        running_loss, total_samples = 0.0, 0
        for image, text, mask, label, event in train_loader:
            image, text, mask, label = image.to(DEVICE), text.to(DEVICE), mask.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                logits = ensemble(image, text, mask)
                loss = criterion(logits, label)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([ensemble.weight_logits, *ensemble.head.parameters()], 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * label.size(0)
            total_samples += label.size(0)
        
        dev_result = evaluate_model(ensemble, dev_loader)
        weights = ensemble.get_weights().detach().cpu().numpy()
        print(f"Epoch {epoch+1:03d} | Loss {running_loss/total_samples:.4f} | DEV Macro-F1 {dev_result['macro_f1']:.4f} | Weights {np.round(weights, 4)}")
        
        if dev_result['macro_f1'] > best_f1:
            best_f1, patience = dev_result['macro_f1'], 0
            best_state = {k: v.detach().cpu().clone() for k, v in ensemble.state_dict().items()}
            print("  ✓ Best model updated.")
        else:
            patience += 1
        if patience >= patience_limit:
            print("\nEarly stopping."); break
            
    ensemble.load_state_dict(best_state)
    ensemble.eval()
    return ensemble, best_f1
