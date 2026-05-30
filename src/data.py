"""
加载并预处理 UCI ISOLET，缓存到 data/isolet.npz 供后续阶段共享。

ISOLET (UCI id=54): 7797 样本、617 维连续特征、26 类（A–Z 字母语音）、无缺失。
预处理: 标签编码到 0..25；StandardScaler 标准化。
约定: 聚类用全量 (X, y)，y 仅作 ARI/NMI 评价不参与聚类；
      AE 训练从 train_idx/val_idx 切（90/10 stratified），val 用于早停。

运行: python -m src.data
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_PATH = DATA_DIR / "isolet.npz"
SEED = 42
VAL_FRACTION = 0.1
UCI_ID = 54  # ISOLET


@dataclass
class Dataset:
    X: np.ndarray              # (n, d) float32, 标准化后
    y: np.ndarray              # (n,)   int64, 标签编码到 [0, n_classes)
    n_classes: int
    feature_names: list[str]
    class_names: list[str]     # 原始类别（编码前）字符串形式
    train_idx: np.ndarray      # AE 训练子集索引
    val_idx: np.ndarray        # AE 验证子集索引

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


# ---------------------------------------------------------------------------

def _fetch_isolet() -> tuple[pd.DataFrame, pd.DataFrame]:
    """从 UCI 拉取 ISOLET。需联网；ucimlrepo 内部会缓存原始 csv。"""
    from ucimlrepo import fetch_ucirepo

    print(f"[fetch] 从 UCI 拉取 ISOLET (id={UCI_ID}) ...")
    bundle = fetch_ucirepo(id=UCI_ID)
    return bundle.data.features, bundle.data.targets


def _check_and_impute(X_df: pd.DataFrame, y_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_missing_X = int(X_df.isna().sum().sum())
    n_missing_y = int(y_df.isna().sum().sum())
    print(f"[check] 缺失值: X={n_missing_X}, y={n_missing_y}")
    if n_missing_X > 0:
        # ISOLET 实际无缺失；保留兜底以适配候选池里其它数据集的换用
        X_df = X_df.fillna(X_df.median(numeric_only=True))
    if n_missing_y > 0:
        y_df = y_df.fillna(y_df.mode().iloc[0])
    return X_df, y_df


def _encode_labels(y_raw: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """把任意类别（int 或 str）映射到 [0, K-1]。"""
    classes_sorted = np.array(sorted(np.unique(y_raw).tolist()))
    class_to_idx = {c: i for i, c in enumerate(classes_sorted.tolist())}
    y = np.asarray([class_to_idx[c] for c in y_raw.tolist()], dtype=np.int64)
    class_names = [str(c) for c in classes_sorted.tolist()]
    return y, class_names


def _make_split(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AE 训练 / 验证切分。stratify 让每类都被覆盖。"""
    train_idx, val_idx = train_test_split(
        np.arange(len(y)),
        test_size=VAL_FRACTION,
        stratify=y,
        random_state=SEED,
    )
    return np.asarray(train_idx), np.asarray(val_idx)


# ---------------------------------------------------------------------------

def load(force_reload: bool = False) -> Dataset:
    """获取预处理后的 ISOLET。若缓存存在则直接读取。"""
    if CACHE_PATH.exists() and not force_reload:
        print(f"[cache] 命中本地缓存: {CACHE_PATH}")
        npz = np.load(CACHE_PATH, allow_pickle=False)
        return Dataset(
            X=npz["X"],
            y=npz["y"],
            n_classes=int(npz["n_classes"].item()),
            feature_names=[str(s) for s in npz["feature_names"]],
            class_names=[str(s) for s in npz["class_names"]],
            train_idx=npz["train_idx"],
            val_idx=npz["val_idx"],
        )

    DATA_DIR.mkdir(exist_ok=True)

    X_df, y_df = _fetch_isolet()
    print(f"[shape] X={tuple(X_df.shape)}, y={tuple(y_df.shape)}")
    X_df, y_df = _check_and_impute(X_df, y_df)

    feature_names = [str(c) for c in X_df.columns.tolist()]
    X_raw = X_df.to_numpy(dtype=np.float32)
    y_raw = y_df.to_numpy().ravel()

    y, class_names = _encode_labels(y_raw)
    n_classes = len(class_names)
    counts = np.bincount(y)
    print(f"[label] 类别数={n_classes}, 每类样本: min={counts.min()}, max={counts.max()}")

    print("[scale] StandardScaler 标准化（zero mean / unit variance per feature）...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)
    assert np.isfinite(X_scaled).all(), "标准化后出现 NaN/Inf"

    train_idx, val_idx = _make_split(y)
    print(f"[split] AE train/val = {len(train_idx)} / {len(val_idx)}")

    print(f"[save] 写入缓存: {CACHE_PATH}")
    np.savez_compressed(
        CACHE_PATH,
        X=X_scaled,
        y=y,
        n_classes=np.int64(n_classes),
        feature_names=np.array(feature_names),
        class_names=np.array(class_names),
        train_idx=train_idx,
        val_idx=val_idx,
    )

    return Dataset(
        X=X_scaled,
        y=y,
        n_classes=n_classes,
        feature_names=feature_names,
        class_names=class_names,
        train_idx=train_idx,
        val_idx=val_idx,
    )


def summary(ds: Dataset) -> str:
    counts = np.bincount(ds.y)
    return (
        "==================== ISOLET ====================\n"
        f"样本数            : {ds.n_samples}\n"
        f"特征数            : {ds.n_features}\n"
        f"类别数 (K)        : {ds.n_classes}\n"
        f"每类样本          : min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}\n"
        f"X 全局 mean / std : {ds.X.mean():+.4e} / {ds.X.std():.4f}\n"
        f"X dtype / 范围    : {ds.X.dtype}, [{ds.X.min():.3f}, {ds.X.max():.3f}]\n"
        f"AE train / val    : {len(ds.train_idx)} / {len(ds.val_idx)}\n"
        "================================================"
    )


if __name__ == "__main__":
    ds = load()
    print()
    print(summary(ds))
