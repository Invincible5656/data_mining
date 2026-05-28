"""
通用聚类流水线：在给定特征空间 X 上跑 BIRCH / OPTICS / Affinity Propagation，
带预测缓存 + ARI/NMI 评价，返回 DataFrame。

cluster_baseline.py / cluster_pca.py / cluster_ae.py 共用本流水线，
区别只在喂进来的 X 是什么（raw / PCA / AE 各隐藏层激活）。

Cache 文件命名: results/predictions/{feature_space}_{algo_slug}.npy
    feature_space 例: raw, pca64, h3
    algo_slug 例: birch, optics_dbscan_p80, ap_pref_p1
    -> raw_birch.npy / pca64_optics_dbscan_p80.npy / h3_ap_pref_p1.npy
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import (
    OPTICS,
    AffinityPropagation,
    Birch,
    cluster_optics_dbscan,
)
from sklearn.metrics.pairwise import euclidean_distances

from .metrics import cluster_stats, evaluate


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "results" / "predictions"
SEED = 42


# --------------------------- 各算法封装 ---------------------------

def run_optics(X: np.ndarray, K: int) -> np.ndarray:
    # 在高维 raw 上 xi 模式总是退化（要么单簇，要么 90% 噪声）。
    # 改用两步法：先让 OPTICS 算 reachability / core_distances，
    # 再用 cluster_optics_dbscan 以一个数据自适应 eps 提取簇。
    # eps 取 80%-tile of core_distances：保证 ~80% 的点能成为 core，
    # 控制噪声比例不失控。
    model = OPTICS(min_samples=5, metric="euclidean", n_jobs=-1)
    model.fit(X)

    finite_core = model.core_distances_[np.isfinite(model.core_distances_)]
    eps = float(np.percentile(finite_core, 80))
    print(f"    [OPTICS] auto eps = 80%-tile of core_distances = {eps:.3f}")

    return cluster_optics_dbscan(
        reachability=model.reachability_,
        core_distances=model.core_distances_,
        ordering=model.ordering_,
        eps=eps,
    )


def run_birch(X: np.ndarray, K: int) -> np.ndarray:
    # 已知真实 K，直接传入；threshold 用默认 0.5
    return Birch(n_clusters=K, threshold=0.5, branching_factor=50).fit_predict(X)


def run_ap(X: np.ndarray, K: int) -> np.ndarray:
    # 取相似度的 1% 分位作为 preference：让大多数样本不做 exemplar，
    # 强制少数高质量样本做聚类中心。
    print("    [AP] 计算相似度矩阵 ...")
    S = (-euclidean_distances(X, X, squared=True)).astype(np.float32)
    n = len(S)
    off_diag = S[~np.eye(n, dtype=bool)]
    preference = float(np.percentile(off_diag, 1))
    del off_diag
    print(f"    [AP] preference = 1%-tile = {preference:.2f}")

    return AffinityPropagation(
        damping=0.9,
        max_iter=300,
        convergence_iter=20,
        preference=preference,
        random_state=SEED,
        affinity="precomputed",
    ).fit_predict(S)


ALGORITHMS: list[tuple[str, str, Callable[[np.ndarray, int], np.ndarray], dict]] = [
    # (display_name, algo_slug, runner, params_for_logging)
    # 注意：超参一变，algo_slug 必须跟着变，否则会读到错的 cache
    ("BIRCH", "birch", run_birch,
     {"n_clusters": "K", "threshold": 0.5, "branching_factor": 50}),
    ("OPTICS", "optics_dbscan_p80", run_optics,
     {"min_samples": 5, "metric": "euclidean",
      "extraction": "cluster_optics_dbscan",
      "eps": "80%-tile of core_distances (auto)"}),
    ("Affinity Propagation", "ap_pref_p1", run_ap,
     {"damping": 0.9, "max_iter": 300, "convergence_iter": 20,
      "preference": "1%-tile of -squared euclidean similarity",
      "affinity": "precomputed"}),
]


@dataclass
class RunResult:
    feature_space: str
    algorithm: str
    n_clusters_target: int
    n_clusters_pred: int
    n_noise: int
    noise_ratio: float
    ari: float
    nmi: float
    seconds: float
    params: str  # JSON


# --------------------------- cache ---------------------------

def _cache_paths(cache_dir: Path, full_slug: str) -> tuple[Path, Path]:
    return cache_dir / f"{full_slug}.npy", cache_dir / f"{full_slug}.json"


def run_or_load(
    cache_dir: Path,
    full_slug: str,
    runner: Callable[[], np.ndarray],
    force: bool,
) -> tuple[np.ndarray, float]:
    pred_p, meta_p = _cache_paths(cache_dir, full_slug)
    if pred_p.exists() and meta_p.exists() and not force:
        meta = json.loads(meta_p.read_text())
        print(f"  [cache] {full_slug}: {pred_p.name}, last run {meta['seconds']:.1f}s")
        return np.load(pred_p), float(meta["seconds"])

    print(f"  [run]   {full_slug} ...")
    t0 = time.time()
    y_pred = runner()
    elapsed = time.time() - t0
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_p, y_pred)
    meta_p.write_text(json.dumps({"seconds": elapsed}))
    print(f"  [done]  {full_slug}: {elapsed:.1f}s")
    return y_pred, elapsed


# --------------------------- pipeline ---------------------------

def run_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    K: int,
    feature_space: str,
    force: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """对 X 跑所有算法 + 评价 + 缓存，返回 DataFrame。CSV 由调用方写。"""
    results: list[RunResult] = []
    for display_name, algo_slug, runner, params in ALGORITHMS:
        full_slug = f"{feature_space}_{algo_slug}"
        print(f">>> [{feature_space}] {display_name}")
        y_pred, secs = run_or_load(
            cache_dir, full_slug, lambda r=runner: r(X, K), force
        )
        m = evaluate(y, y_pred)
        s = cluster_stats(y_pred)
        rr = RunResult(
            feature_space=feature_space,
            algorithm=display_name,
            n_clusters_target=K,
            n_clusters_pred=int(s["n_clusters_pred"]),
            n_noise=int(s["n_noise"]),
            noise_ratio=float(s["noise_ratio"]),
            ari=m["ARI"],
            nmi=m["NMI"],
            seconds=float(secs),
            params=json.dumps(params),
        )
        results.append(rr)
        print(
            f"  -> ARI={rr.ari:+.4f}  NMI={rr.nmi:.4f}  "
            f"clusters={rr.n_clusters_pred}/{K}  "
            f"noise={rr.n_noise} ({rr.noise_ratio:.1%})"
        )
        print()

    return pd.DataFrame([asdict(r) for r in results])
