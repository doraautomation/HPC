import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# -----------------------------
# MLP Router
# -----------------------------

class ShardRouterMLP(nn.Module):
    def __init__(self, input_dim, num_shards, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_shards)
        )
    def forward(self, x):
        return self.net(x)


# -----------------------------
# Build training data
# -----------------------------

def build_training_data_from_shards(shards):
    all_X, all_y = [], []
    for F_i, L_i in shards:
        all_X.append(F_i)
        all_y.append(L_i)
    return np.concatenate(all_X), np.concatenate(all_y)


# -----------------------------
# Train MLP router (offline)
# -----------------------------

def train_mlp_router_from_shards(shards, input_dim, num_shards,
                                 batch_size=256, lr=1e-3, epochs=10,
                                 device="cuda"):
    X_np, y_np = build_training_data_from_shards(shards)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    model = ShardRouterMLP(input_dim, num_shards).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()
    return model


# -----------------------------
# Aging weight
# -----------------------------

def compute_aging_weight(N_j, T, beta):
    return 1.0 / (1 + np.exp(-beta * (N_j - T)))


# -----------------------------
#  Online insertion
# -----------------------------

def online_insert_with_mlp_routing(
    v_fuse, centroids, shard_counts, mlp_model,
    alpha, T, beta, device="cuda"
):
    k, d = centroids.shape
    v_fuse = v_fuse.astype(np.float32)
    centroids = centroids.astype(np.float32)

    # Step 1: MLP prediction
    mlp_model.eval()
    with torch.no_grad():
        x = torch.tensor(v_fuse).unsqueeze(0).to(device)
        probs = torch.softmax(mlp_model(x), dim=1).cpu().numpy()[0]

    # Step 2–3: Aging weights
    W = np.array([compute_aging_weight(shard_counts[j], T, beta) for j in range(k)])

    # Step 4: Initial shard from MLP
    k_mlp = int(np.argmax(probs))

    # Step 5: Aging-adjusted refinement
    def dist(a, b): return np.linalg.norm(a - b)
    k_star = k_mlp
    d_best = dist(v_fuse, centroids[k_mlp]) * W[k_mlp]

    for j in range(k):
        if j == k_mlp:
            continue
        d_j = dist(v_fuse, centroids[j]) * W[j]
        if d_j < d_best:
            d_best = d_j
            k_star = j

    # Step 6: Drift-controlled update
    eta = 1.0 / (shard_counts[k_star] + alpha)
    centroids[k_star] += eta * (v_fuse - centroids[k_star])

    # Step 7: Update shard size
    shard_counts[k_star] += 1

    return k_star, centroids, shard_counts
