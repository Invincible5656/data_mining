"""
Part 1 (raw 特征空间): 在原始 617 维标准化特征上跑 BIRCH / OPTICS / Affinity Propagation。

实际逻辑都在 pipeline.run_pipeline 里，本文件是 feature_space="raw" 的薄包装。

运行: python -m src.cluster_baseline [--force]
"""

from __future__ import annotations

import argparse

from . import data
from .pipeline import PROJECT_ROOT, run_pipeline


CSV_PATH = PROJECT_ROOT / "results" / "clustering_raw.csv"


def main(force: bool = False):
    ds = data.load()
    print(data.summary(ds))
    print()

    df = run_pipeline(ds.X, ds.y, ds.n_classes, feature_space="raw", force=force)

    df.to_csv(CSV_PATH, index=False)
    print(f"[save] {CSV_PATH}")
    print()
    print(df[["algorithm", "n_clusters_pred", "noise_ratio",
              "ari", "nmi", "seconds"]].to_string(index=False))
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="忽略缓存，所有算法都重跑")
    args = p.parse_args()
    main(force=args.force)
