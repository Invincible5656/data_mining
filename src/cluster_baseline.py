"""
Part 1: 原始 617 维标准化特征上跑选中的 3 个聚类算法。
        OPTICS / BIRCH / Affinity Propagation
        记录 ARI / NMI / 簇数 / 噪声 / 时间。

设计要点:
    - 每个算法的预测标签缓存到 results/baseline_predictions/，避免反复重跑；
      想强制重跑用 `python -m src.cluster_baseline --force`。
    - 真实 K=26，但 OPTICS / AP 不输入 K，由算法自行决定（记录最终簇数）。
    - 最终汇总写到 results/baseline_clustering.csv。

性能预估（macOS, 7797 × 617）:
    BIRCH               ~ 几秒
    OPTICS              ~ 5-20 分钟（min_samples=10）
    AffinityPropagation ~ 10-30 分钟，峰值内存 ~1 GB

运行: python -m src.cluster_baseline [--force]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS, AffinityPropagation, Birch

from . import data
from .metrics import cluster_stats, evaluate


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PREDS_DIR = RESULTS_DIR / "baseline_predictions"
CSV_PATH = RESULTS_DIR / "baseline_clustering.csv"
SEED = 42


# --------------------------- 各算法封装 ---------------------------

def run_optics(X: np.ndarray, K: int) -> np.ndarray:
    # min_samples=10：高维下默认 5 太小、噪声会过多
    # cluster_method="xi"：变密度更稳健，比 "dbscan" 模式更通用
    model = OPTICS(min_samples=10, metric="euclidean",
                   cluster_method="xi", n_jobs=-1)
    model.fit(X)
    return model.labels_


def run_birch(X: np.ndarray, K: int) -> np.ndarray:
    # 已知真实 K=26，直接传入；threshold 用默认 0.5
    # 若 sub-cluster 过多可调大 threshold 或减小 branching_factor
    model = Birch(n_clusters=K, threshold=0.5, branching_factor=50)
    return model.fit_predict(X)


def run_ap(X: np.ndarray, K: int) -> np.ndarray:
    # damping=0.9 提高数值稳定性，max_iter / convergence_iter 给慢收敛余量
    # preference=None -> 用相似度中位数，簇数由算法自行决定
    model = AffinityPropagation(
        damping=0.9, max_iter=200, convergence_iter=15, random_state=SEED
    )
    return model.fit_predict(X)


ALGORITHMS: list[tuple[str, str, Callable[[np.ndarray, int], np.ndarray], dict]] = [
    # (display_name, cache_slug, runner, params_for_logging)
    # cache slug 里编了关键超参版本号，超参一改 slug 就变，老 cache 自动失效
    ("BIRCH", "birch", run_birch,
     {"n_clusters": "K", "threshold": 0.5, "branching_factor": 50}),
    ("OPTICS", "optics", run_optics,
     {"min_samples": 10, "metric": "euclidean", "cluster_method": "xi"}),
    ("Affinity Propagation", "ap", run_ap,
     {"damping": 0.9, "max_iter": 200, "convergence_iter": 15}),
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


# --------------------------- 缓存 ---------------------------

def _cache_paths(slug: str) -> tuple[Path, Path]:
    pred_p = PREDS_DIR / f"raw_{slug}.npy"
    meta_p = PREDS_DIR / f"raw_{slug}.json"
    return pred_p, meta_p


def run_or_load(slug: str, runner: Callable[[], np.ndarray],
                force: bool) -> tuple[np.ndarray, float]:
    pred_p, meta_p = _cache_paths(slug)
    if pred_p.exists() and meta_p.exists() and not force:
        meta = json.loads(meta_p.read_text())
        print(f"  [cache] {slug}: {pred_p.name}, last run {meta['seconds']:.1f}s")
        return np.load(pred_p), float(meta["seconds"])

    print(f"  [run]   {slug} ...")
    t0 = time.time()
    y_pred = runner()
    elapsed = time.time() - t0
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(pred_p, y_pred)
    meta_p.write_text(json.dumps({"seconds": elapsed}))
    print(f"  [done]  {slug}: {elapsed:.1f}s")
    return y_pred, elapsed


# --------------------------- 主流程 ---------------------------

def main(force: bool = False) -> pd.DataFrame:
    ds = data.load()
    print(data.summary(ds))
    print()

    K, X, y = ds.n_classes, ds.X, ds.y
    results: list[RunResult] = []

    for display_name, slug, runner, params in ALGORITHMS:
        print(f">>> {display_name}")
        y_pred, secs = run_or_load(slug, lambda r=runner: r(X, K), force=force)
        m = evaluate(y, y_pred)
        s = cluster_stats(y_pred)
        rr = RunResult(
            feature_space="raw",
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

    df = pd.DataFrame([asdict(r) for r in results])
    RESULTS_DIR.mkdir(exist_ok=True)
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
