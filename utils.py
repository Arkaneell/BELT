import os
import random
import numpy as np
import torch
import re
import html

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_text(text):
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'([!?.,])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_dataframe(df, text_column='description'):
    df = df.copy()
    before = len(df)
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    valid = df[text_column].str.strip().str.len() > 0
    df = df[valid].reset_index(drop=True)
    print(f"Rows: {before} -> {len(df)}")
    return df

def extract_event(filename):
    parts = str(filename).split('/')
    return parts[1] if len(parts) >= 2 else 'unknown'
