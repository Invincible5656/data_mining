"""
对称欠完备自编码器 (PyTorch)。

架构: I=617 - H1=256 - H2=128 - H3=64 - H4=128 - H5=256 - O=617
    - H3 是 bottleneck，对应 PCA-64
    - H1/H5 对应 PCA-256，H2/H4 对应 PCA-128
    - 隐层全部 ReLU；输出层不加激活（输入是 StandardScaler 标准化后的，可正可负）

训练:
    - 损失   : MSE(reconstruction, input)
    - 优化器 : Adam, lr=1e-3, weight_decay=0
    - batch  : 256
    - epochs : 最多 200，patience=15 早停（看 val MSE）
    - 数据切分: 复用 data.py 里的 train_idx / val_idx (90/10 stratified)
    - 设备   : MPS (Apple Silicon) -> CUDA -> CPU 自动选

产出:
    - models/ae.pt       : 训练完的 state_dict + 架构元信息
    - results/ae_loss.csv: 每个 epoch 的 train/val loss

接口:
    Autoencoder.encode_layers(X) -> dict[name -> activation]
        返回 {"h1": ..., "h2": ..., "h3": ..., "h4": ..., "h5": ...}
        供 cluster_ae.py 喂给 pipeline.run_pipeline。

运行: python -m src.autoencoder [--force]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import data
from .pipeline import PROJECT_ROOT, SEED


MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "ae.pt"
LOSS_CSV = PROJECT_ROOT / "results" / "ae_loss.csv"

HIDDEN_DIMS = (256, 128, 64, 128, 256)  # H1..H5
BATCH_SIZE = 256
LR = 1e-3
MAX_EPOCHS = 200
PATIENCE = 15


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Autoencoder(nn.Module):
    """5 隐层对称 AE。每层都是 Linear，hidden 用 ReLU，输出层不加激活。"""

    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...] = HIDDEN_DIMS):
        super().__init__()
        assert len(hidden_dims) == 5, "本实验固定 5 个隐层"
        h1, h2, h3, h4, h5 = hidden_dims
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims

        # 每个 fc 后面都跟 ReLU，最后 fc_out 不加激活
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.fc5 = nn.Linear(h4, h5)
        self.fc_out = nn.Linear(h5, in_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        h3 = self.act(self.fc3(h2))
        h4 = self.act(self.fc4(h3))
        h5 = self.act(self.fc5(h4))
        return self.fc_out(h5)

    @torch.no_grad()
    def encode_layers(self, X: np.ndarray, device: torch.device,
                      batch_size: int = 1024) -> dict[str, np.ndarray]:
        """对全量 X 跑一次前向，收集 h1..h5 激活，返回 numpy。"""
        self.eval()
        outs: dict[str, list[np.ndarray]] = {f"h{i}": [] for i in range(1, 6)}
        n = len(X)
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            h1 = self.act(self.fc1(xb))
            h2 = self.act(self.fc2(h1))
            h3 = self.act(self.fc3(h2))
            h4 = self.act(self.fc4(h3))
            h5 = self.act(self.fc5(h4))
            outs["h1"].append(h1.cpu().numpy())
            outs["h2"].append(h2.cpu().numpy())
            outs["h3"].append(h3.cpu().numpy())
            outs["h4"].append(h4.cpu().numpy())
            outs["h5"].append(h5.cpu().numpy())
        return {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in outs.items()}


def train(force: bool = False) -> Autoencoder:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = pick_device()
    print(f"[device] {device}")

    ds = data.load()
    print(data.summary(ds))
    print()

    if MODEL_PATH.exists() and not force:
        print(f"[cache] 命中 {MODEL_PATH}, 直接加载。--force 可重训。")
        model = Autoencoder(in_dim=ds.n_features)
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        return model

    X = torch.from_numpy(ds.X)
    X_train = X[ds.train_idx]
    X_val = X[ds.val_idx]
    print(f"[data] train={X_train.shape}, val={X_val.shape}")

    train_loader = DataLoader(
        TensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=False,
    )
    X_val_dev = X_val.to(device)

    model = Autoencoder(in_dim=ds.n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state: dict | None = None
    bad_epochs = 0
    history: list[dict] = []

    print(f"[train] max_epochs={MAX_EPOCHS}, batch={BATCH_SIZE}, lr={LR}, patience={PATIENCE}")
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum, n_seen = 0.0, 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            recon = model(xb)
            loss = loss_fn(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = xb.size(0)
            train_loss_sum += loss.item() * bs
            n_seen += bs
        train_loss = train_loss_sum / n_seen

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_dev), X_val_dev).item()

        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        marker = "*" if improved else " "
        print(f"  epoch {epoch:3d} | train {train_loss:.5f} | val {val_loss:.5f} {marker}  "
              f"(bad={bad_epochs}/{PATIENCE})")

        if bad_epochs >= PATIENCE:
            print(f"[early-stop] val 连续 {PATIENCE} epoch 没改进，停。")
            break

    elapsed = time.time() - t0
    print(f"[done] best val MSE = {best_val:.5f}, 用时 {elapsed:.1f}s")

    assert best_state is not None
    model.load_state_dict(best_state)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "in_dim": ds.n_features,
            "hidden_dims": list(HIDDEN_DIMS),
            "best_val_mse": best_val,
            "epochs_trained": len(history),
        },
        MODEL_PATH,
    )
    print(f"[save] {MODEL_PATH}")

    pd.DataFrame(history).to_csv(LOSS_CSV, index=False)
    print(f"[save] {LOSS_CSV}")

    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="忽略已存在的 ae.pt，重新训练")
    args = p.parse_args()
    train(force=args.force)
