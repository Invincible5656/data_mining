"""
用 Word Embedding 在姓名拼音与候选算法/数据集名之间算 L2 距离，
选出最近的 3 个聚类算法 + 1 个 UCI 数据集，结果写入 results/selection.json。

模型: paraphrase-multilingual-MiniLM-L12-v2（多语言，拼音与英文术语落到同一空间）。
距离: 原始 embedding 上的 L2（不做归一化）。

运行: python -m src.select
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


USER_NAME_PINYIN = "lanjiajun"

CLUSTERING_ALGORITHMS = [
    "Affinity Propagation",
    "BIRCH",
    "DBSCAN",
    "Hierarchical Clustering",
    "K-Means",
    "Mean Shift",
    "OPTICS",
    "Spectral Clustering",
]

# UCI 分类数据集候选池（样本数 ≥ 5000）。
UCI_CLASSIFICATION_DATASETS = [
    "Adult",
    "Bank Marketing",
    "Default of Credit Card Clients",
    "Letter Recognition",
    "MAGIC Gamma Telescope",
    "Pen-Based Recognition of Handwritten Digits",
    "Optical Recognition of Handwritten Digits",
    "Mushroom",
    "Nursery",
    "Statlog Shuttle",
    "Avila",
    "Crowdsourced Mapping",
    "Online Shoppers Purchasing Intention",
    "Polish Companies Bankruptcy",
    "Skin Segmentation",
    "Connect-4",
    "Chess King-Rook vs King",
    "Covertype",
    "HTRU2",
    "Sensorless Drive Diagnosis",
    "EEG Eye State",
    "Anuran Calls MFCCs",
    "Page Blocks Classification",
    "ISOLET",
    "Statlog Landsat Satellite",
    "Waveform Database Generator Version 2",
    "Wall-Following Robot Navigation",
    "Gas Sensor Array Drift",
    "Human Activity Recognition Using Smartphones",
    "Smartphone-Based Recognition of Human Activities",
    "Online News Popularity",
    "Gesture Phase Segmentation",
    "Codon Usage",
    "Gisette",
    "MoCap Hand Postures",
    "Census-Income KDD",
    "Frogs Anuran Calls",
    "Firm-Teacher Clave-Direction Classification",
    "Diabetes 130 US Hospitals",
    "Dry Bean Dataset",
    "HCV Data",
    "Localization Data for Person Activity",
]

# 多语言句子嵌入模型；首次运行会从 HuggingFace 下载约 500MB
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K_ALGORITHMS = 3
TOP_K_DATASETS = 1


# ----------------------------- 实现 -----------------------------

def l2_distances(query_vec: np.ndarray, candidate_vecs: np.ndarray) -> np.ndarray:
    """query (1, d) 到每个 candidate (n, d) 的欧氏距离，返回 (n,)"""
    return np.linalg.norm(candidate_vecs - query_vec, axis=1)


def encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return np.asarray(
        model.encode(texts, normalize_embeddings=False, show_progress_bar=False),
        dtype=np.float32,
    )


def select() -> dict:
    print(f"[1/3] 加载 embedding 模型: {MODEL_NAME}")
    print("      （首次运行从 HuggingFace 下载约 500MB，之后命中本地缓存）")
    model = SentenceTransformer(MODEL_NAME)

    print(f"[2/3] 编码姓名 / 算法 / 数据集")
    name_vec = encode(model, [USER_NAME_PINYIN])
    algo_vecs = encode(model, CLUSTERING_ALGORITHMS)
    ds_vecs = encode(model, UCI_CLASSIFICATION_DATASETS)

    algo_dists = l2_distances(name_vec, algo_vecs)
    ds_dists = l2_distances(name_vec, ds_vecs)

    algo_ranked = sorted(
        zip(CLUSTERING_ALGORITHMS, algo_dists.tolist()),
        key=lambda x: x[1],
    )
    ds_ranked = sorted(
        zip(UCI_CLASSIFICATION_DATASETS, ds_dists.tolist()),
        key=lambda x: x[1],
    )

    selected_algos = algo_ranked[:TOP_K_ALGORITHMS]
    selected_datasets = ds_ranked[:TOP_K_DATASETS]

    result = {
        "user_name_pinyin": USER_NAME_PINYIN,
        "embedding_model": MODEL_NAME,
        "distance_metric": "L2 (Euclidean) on raw embeddings",
        "selected_algorithms": [
            {"name": n, "l2_distance": d} for n, d in selected_algos
        ],
        "selected_dataset": {
            "name": selected_datasets[0][0],
            "l2_distance": selected_datasets[0][1],
        },
        "all_algorithm_distances": [
            {"name": n, "l2_distance": d} for n, d in algo_ranked
        ],
        "all_dataset_distances": [
            {"name": n, "l2_distance": d} for n, d in ds_ranked
        ],
    }

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "selection.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[3/3] 写入 {out_path}")
    print()
    print(f"姓名拼音: {USER_NAME_PINYIN}")
    print(f"--- 选中的 {TOP_K_ALGORITHMS} 个聚类算法 ---")
    for n, d in selected_algos:
        print(f"  {n:<28}  L2 = {d:.4f}")
    print(f"--- 选中的数据集 ---")
    for n, d in selected_datasets:
        print(f"  {n:<28}  L2 = {d:.4f}")
    print()
    print("（完整距离表见 selection.json）")
    return result


if __name__ == "__main__":
    select()
