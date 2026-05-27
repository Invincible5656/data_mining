"""
聚类评价指标封装：ARI / NMI + 簇数 / 噪声点统计。

ARI: Adjusted Rand Index, 越接近 1 越好，随机划分约 0
NMI: Normalized Mutual Information, [0, 1]，越接近 1 越好
噪声点: 部分算法（DBSCAN / OPTICS）会输出 label = -1 表示噪声，
        sklearn 的 ARI/NMI 会把所有 -1 当作同一个簇 —— 这会让分数虚高，
        因此另外报告噪声比例供报告里说明处理方式。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(
            normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")
        ),
    }


def cluster_stats(y_pred: np.ndarray) -> dict[str, float | int]:
    unique, counts = np.unique(y_pred, return_counts=True)
    has_noise = -1 in unique.tolist()
    n_noise = int(counts[unique == -1].sum()) if has_noise else 0
    n_clusters = int(len(unique) - (1 if has_noise else 0))
    return {
        "n_clusters_pred": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": float(n_noise / max(len(y_pred), 1)),
    }
