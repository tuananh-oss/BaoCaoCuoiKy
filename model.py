"""Candidate generator DNN (Hinh 2, Bang 3 cua bai bao).

Embedding 64-d -> trung binh (bo qua padding) -> FC 1024 -> 512 -> 256
(moi lop: GroupNorm(32) + ReLU) -> FC n_movies -> softmax (trong CrossEntropyLoss).
FL dung GroupNorm (khong co running stats); non-FL dung BatchNorm.
"""
import torch
import torch.nn as nn


class CandidateGenerator(nn.Module):
    def __init__(self, n_movies, emb_dim=64, hidden=(1024, 512, 256), norm="group"):
        super().__init__()
        self.emb = nn.Embedding(n_movies, emb_dim, padding_idx=0)
        layers, d = [], emb_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.GroupNorm(32, h) if norm == "group" else nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            d = h
        self.tower = nn.Sequential(*layers)
        self.out = nn.Linear(d, n_movies)

    def forward(self, x):                      # x: [B, WINDOW]
        e = self.emb(x)                        # [B, W, D]
        mask = (x != 0).unsqueeze(-1).float()
        avg = (e * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.out(self.tower(avg))       # logits


@torch.no_grad()
def top_k_accuracy(model, X, y, k=100, batch=2048, device="cpu"):
    model.eval()
    hit, n = 0, len(y)
    for i in range(0, n, batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        yb = torch.from_numpy(y[i:i + batch]).to(device)
        logits = model(xb)
        logits[:, 0] = -1e9                    # khong du doan padding
        topk = logits.topk(k, dim=1).indices
        hit += (topk == yb.unsqueeze(1)).any(1).sum().item()
    return hit / n
