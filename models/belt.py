import torch
from torch import nn
import torch.nn.functional as F
from config import TEXT_DIM, HIDDEN_DIM, NUM_HEADS, DROPOUT, EMBED_DIM, NUM_CLASSES
from models.coral import CoralHead

class AdapterPoolText(nn.Module):
    def __init__(self, input_dim=TEXT_DIM, output_dim=HIDDEN_DIM, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(input_dim, bottleneck)
        self.up = nn.Linear(bottleneck, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, text, mask):
        residual = text
        x = F.gelu(self.down(text))
        x = self.up(x)
        x = self.norm(residual + x)
        mask = mask.unsqueeze(-1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.projection(x)

class BELT(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_projection = nn.Sequential(
            nn.Linear(1280, HIDDEN_DIM), # Use 1280 explicitly or from config
            nn.LayerNorm(HIDDEN_DIM), nn.ReLU()
        )
        self.text_projection = AdapterPoolText()
        self.image_to_text = nn.MultiheadAttention(HIDDEN_DIM, NUM_HEADS, dropout=DROPOUT, batch_first=True)
        self.text_to_image = nn.MultiheadAttention(HIDDEN_DIM, NUM_HEADS, dropout=DROPOUT, batch_first=True)
        self.image_norm = nn.LayerNorm(HIDDEN_DIM)
        self.text_norm = nn.LayerNorm(HIDDEN_DIM)
        self.self_attention = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, nhead=NUM_HEADS, dim_feedforward=512,
            dropout=DROPOUT, batch_first=True, norm_first=True
        )
        self.embedding = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM), nn.ReLU()
        )
        self.coral_head = CoralHead(EMBED_DIM, num_classes=NUM_CLASSES)

    def embed(self, image, text, mask):
        if image.ndim == 4:
            image = F.adaptive_avg_pool2d(image, 1).flatten(1)
        v = self.image_projection(image)
        t = self.text_projection(text, mask)
        v_tok, t_tok = v.unsqueeze(1), t.unsqueeze(1)
        v_att, _ = self.image_to_text(v_tok, t_tok, t_tok)
        t_att, _ = self.text_to_image(t_tok, v_tok, v_tok)
        v_new = self.image_norm(v_tok + v_att)
        t_new = self.text_norm(t_tok + t_att)
        pair = torch.cat([v_new, t_new], dim=1)
        pair = self.self_attention(pair)
        fused = pair.flatten(start_dim=1)
        return self.embedding(fused)

    def forward(self, image, text, mask):
        return self.coral_head(self.embed(image, text, mask))
