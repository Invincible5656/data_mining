"""
Part 2 (AE 特征空间): 在训好的自编码器每个隐层 (H1..H5) 上跑 BIRCH / OPTICS / Affinity Propagation。

H1: 256d        ↔ PCA-256
H2: 128d        ↔ PCA-128
H3: 64d (bottleneck) ↔ PCA-64
H4: 128d        ↔ PCA-128
H5: 256d        ↔ PCA-256

同维严格对比：H3 vs PCA-64、H2/H4 vs PCA-128、H1/H5 vs PCA-256，验证非线性是否额外贡献。

运行: python -m src.cluster_ae [--force]
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import data
from .autoencoder import pick_device, train as ae_train
from .pipeline import PROJECT_ROOT, run_pipeline


CSV_PATH = PROJECT_ROOT / "results" / "clustering_ae.csv"

LAYER_NAMES = ["h1", "h2", "h3", "h4", "h5"]


def main(force: bool = False) -> pd.DataFrame:
    ds = data.load()
    print(data.summary(ds))
    print()

    # 加载（或训练）AE
    model = ae_train(force=False)  # AE 模型本身不重训；只重跑聚类
    device = pick_device()
    model.to(device)

    print("[encode] 抽取 H1..H5 激活 ...")
    feats = model.encode_layers(ds.X, device=device)
    for name in LAYER_NAMES:
        Z = feats[name]
        sparsity = float((Z == 0).mean())
        print(f"  {name}: shape={Z.shape}, sparsity(=0)={sparsity:.1%}, "
              f"mean={Z.mean():+.3f}, std={Z.std():.3f}")
    print()

    dfs: list[pd.DataFrame] = []
    for name in LAYER_NAMES:
        print(f"================ AE layer {name} (dim={feats[name].shape[1]}) ================")
        df = run_pipeline(
            feats[name], ds.y, ds.n_classes,
            feature_space=name,
            force=force,
        )
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)
    full.to_csv(CSV_PATH, index=False)
    print(f"[save] {CSV_PATH}")
    print()
    print(
        full[["feature_space", "algorithm", "n_clusters_pred",
              "noise_ratio", "ari", "nmi", "seconds"]].to_string(index=False)
    )
    return full


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="忽略缓存，所有 AE 层 × 算法都重跑")
    args = p.parse_args()
    main(force=args.force)
