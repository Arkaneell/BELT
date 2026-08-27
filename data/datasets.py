import torch
import numpy as np
from torch.utils.data import Dataset
from config import DEVICE, NUM_CLASSES

class InformDataset(Dataset):
    def __init__(self, data, event2idx):
        self.images = torch.from_numpy(data['img_maps']).float()
        self.text = torch.from_numpy(data['txt_seqs']).float()
        self.masks = torch.from_numpy(data['txt_masks']).float()
        self.labels = torch.from_numpy(data['labels']).long()
        self.events = torch.tensor([event2idx[e] for e in data['events']], dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, index):
        return self.images[index], self.text[index], self.masks[index], self.labels[index], self.events[index]

def get_class_weights(train_labels):
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float32)
    weights = 1.0 / np.sqrt(class_counts)
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

def balanced_bootstrap_sample(class_pools, samples_per_class, seed):
    import numpy as np
    rng = np.random.RandomState(seed)
    indices = []
    for class_id in range(NUM_CLASSES):
        sampled = rng.choice(class_pools[class_id], size=samples_per_class, replace=True)
        indices.extend(sampled.tolist())
    rng.shuffle(indices)
    return indices
