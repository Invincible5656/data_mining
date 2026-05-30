"""
PCA 对照实验：把 ISOLET 标准化特征降到 {64, 128, 256}，各维度跑
BIRCH / OPTICS / Affinity Propagation，结果写到 results/clustering_pca.csv。

维度对齐 AE 隐层 (PCA-256↔H1/H5, PCA-128↔H2/H4, PCA-64↔H3 瓶颈)，
用于 PCA-d vs AE-d 同维对比，验证非线性是否带来额外提升。

运行: python -m src.cluster_pca [--force]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import data
from .pipeline import PROJECT_ROOT, SEED, run_pipeline


CSV_PATH = PROJECT_ROOT / "results" / "clustering_pca.csv"

# 与 AE 架构对齐：(256, 128, 64, 128, 256) 的 unique 维度
PCA_DIMS = [64, 128, 256]


def main(force: bool = False) -> pd.DataFrame:
    ds = data.load()
    print(data.summary(ds))
    print()

    dfs: list[pd.DataFrame] = []
    for n_components in PCA_DIMS:
        print(f"================ PCA n_components = {n_components} ================")
        pca = PCA(n_components=n_components, random_state=SEED)
        Z = pca.fit_transform(ds.X).astype(np.float32)
        evr = float(pca.explained_variance_ratio_.sum())
        print(f"  Z shape = {Z.shape}")
        print(f"  explained variance ratio = {evr:.4f}")
        print()

        df = run_pipeline(
            Z, ds.y, ds.n_classes,
            feature_space=f"pca{n_components}",
            force=force,
        )
        df["pca_explained_variance"] = evr
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
                   help="忽略缓存，所有 PCA 维度 × 算法都重跑")
    args = p.parse_args()
    main(force=args.force)
