import torch
from torch import nn
from models.coral import CoralHead
from config import EMBED_DIM, NUM_CLASSES

class EmbeddingEnsemble(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.num_members = len(members)
        self.weight_logits = nn.Parameter(torch.zeros(self.num_members))
        self.head = CoralHead(EMBED_DIM, num_classes=NUM_CLASSES)

    def get_weights(self):
        return torch.softmax(self.weight_logits, dim=0)

    def get_member_embeddings(self, image, text, mask):
        embeddings = []
        with torch.no_grad():
            for member in self.members:
                embeddings.append(member.embed(image, text, mask))
        return torch.stack(embeddings, dim=0)

    def forward(self, image, text, mask):
        embeddings = self.get_member_embeddings(image, text, mask)
        weights = self.get_weights().view(-1, 1, 1)
        ensemble_embedding = (embeddings * weights).sum(dim=0)
        return self.head(ensemble_embedding)
