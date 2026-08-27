import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from transformers import AutoTokenizer, AutoModel
from config import BASE, IMG_ROOT, DEVICE, USE_AMP, MAX_TEXT_LENGTH

TEXT_MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"

def get_feature_extractors():
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(DEVICE).eval()
    for p in text_model.parameters(): p.requires_grad = False

    image_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).features.to(DEVICE).eval()
    for p in image_model.parameters(): p.requires_grad = False

    image_transform = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return tokenizer, text_model, image_model, image_transform

class RawImageTextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, image_transform):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_transform = image_transform

    def __len__(self): return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(os.path.join(IMG_ROOT, row['filename'])).convert('RGB')
        image = self.image_transform(image)
        encoded = self.tokenizer(str(row['description']), truncation=True, padding='max_length',
                                 max_length=MAX_TEXT_LENGTH, return_tensors='pt')
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        label = int(row['text_inform'])
        
        # Use local import to avoid circularity if needed, but extract_event is in utils
        from utils import extract_event
        event = extract_event(row['filename'])
        
        return image, input_ids, attention_mask, label, event

def extract_and_cache(dataframe, cache_path, batch_size=128, num_workers=2):
    if os.path.exists(cache_path):
        print("Loading:", cache_path)
        with open(cache_path, 'rb') as file:
            return pickle.load(file)

    tokenizer, text_model, image_model, image_transform = get_feature_extractors()
    dataset = RawImageTextDataset(dataframe, tokenizer, image_transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    image_features, text_features, text_masks, labels, events = [], [], [], [], []
    with torch.no_grad():
        for images, input_ids, attn_masks, batch_labels, batch_events in tqdm(loader, desc=os.path.basename(cache_path)):
            images = images.to(DEVICE, non_blocking=True)
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            attn_masks = attn_masks.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                img_feat = image_model(images)
                txt_out = text_model(input_ids=input_ids, attention_mask=attn_masks).last_hidden_state
            image_features.append(img_feat.cpu().numpy().astype(np.float16))
            text_features.append(txt_out.cpu().numpy().astype(np.float16))
            text_masks.append(attn_masks.cpu().numpy().astype(np.int64))
            labels.extend(batch_labels)
            events.extend(batch_events)

    data = {
        'img_maps': np.concatenate(image_features, axis=0),
        'txt_seqs': np.concatenate(text_features, axis=0),
        'txt_masks': np.concatenate(text_masks, axis=0),
        'labels': np.array(labels, dtype=np.int64),
        'events': events
    }
    with open(cache_path, 'wb') as file:
        pickle.dump(data, file)
    print(f"Saved {cache_path}: {os.path.getsize(cache_path)/1024**2:.2f} MB")
    return data
