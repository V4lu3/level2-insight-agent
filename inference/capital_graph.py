# -*- coding: utf-8 -*-
"""
参与者语义资金拓扑图构建器 (Capital Topology Network)
=====================================================
根据各簇质心特征向量计算余弦相似度连边，构建全市场参与者原型关系网络，生成拓扑 JSON 与高分辨率可视化图片。
"""
import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

logger = logging.getLogger(__name__)


def build(centroids: dict, labels: dict, out_dir: str, sim_thresh: float = 0.50):
    """
    centroids: {cid: {feature: val}}
    labels: {cid: pattern_type_string}
    """
    c_ids = sorted(list(centroids.keys()))
    if not c_ids:
        return None, None
        
    feat_names = list(centroids[c_ids[0]].keys())
    mat = np.array([[centroids[cid][f] for f in feat_names] for cid in c_ids])
    
    # 归一化向量余弦相似度
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    norm_mat = mat / norms
    sim_mat = np.dot(norm_mat, norm_mat.T)

    G = nx.Graph()
    for i, cid in enumerate(c_ids):
        G.add_node(cid, label=labels.get(cid, f"Cluster_{cid}"))
        
    for i in range(len(c_ids)):
        for j in range(i + 1, len(c_ids)):
            sim = float(sim_mat[i, j])
            if sim >= sim_thresh:
                G.add_edge(c_ids[i], c_ids[j], weight=round(sim, 3))

    os.makedirs(out_dir, exist_ok=True)
    
    # 导出 JSON
    nodes_data = [{"id": int(n), "label": G.nodes[n]["label"]} for n in G.nodes]
    edges_data = [{"source": int(u), "target": int(v), "weight": G.edges[u, v]["weight"]} for u, v in G.edges]
    graph_json_path = os.path.join(out_dir, "capital_graph.json")
    with open(graph_json_path, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes_data, "edges": edges_data}, f, ensure_ascii=False, indent=2)

    # 导出高分辨率图片
    plt.figure(figsize=(8, 6), dpi=150)
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color="skyblue", alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6, edge_color="gray")
    node_labels = {n: f"{n}:{G.nodes[n]['label'][:4]}" for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    
    plt.title("AMI Capital Microstructure Topology Graph", fontsize=12)
    plt.axis("off")
    graph_img_path = os.path.join(out_dir, "capital_graph.png")
    plt.savefig(graph_img_path, bbox_inches="tight")
    plt.close()

    logger.info(f"资金拓扑图构建完成: Node={len(G.nodes)}, Edge={len(G.edges)} -> {graph_img_path}")
    return G, graph_img_path
