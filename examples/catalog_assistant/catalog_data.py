"""同时加载 mlblack + nsgablack catalog"""
import sys
import os
from typing import List, Dict, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_NSGABLACK_ROOT = r"C:\Users\hp\Desktop\nsgablack"

for _p in (_PROJECT_ROOT, _NSGABLACK_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_catalog_cache = None


def _load_catalog():
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache

    all_entries = []

    # mlblack
    try:
        from mlblack.catalog.registry import get_catalog
        for e in get_catalog(refresh=False).list():
            all_entries.append(_entry_to_dict(e, "mlblack"))
    except Exception as e:
        print(f"[catalog] mlblack 加载失败: {e}")

    # nsgablack
    try:
        from nsgablack.catalog.registry import get_catalog
        for e in get_catalog(refresh=False).list():
            all_entries.append(_entry_to_dict(e, "nsgablack"))
    except Exception as e:
        print(f"[catalog] nsgablack 加载失败: {e}")

    _catalog_cache = all_entries
    kinds = sorted(set(e["kind"] for e in all_entries))
    print(f"[catalog] 加载完成: {len(all_entries)} 组件, 类别: {kinds}")
    return all_entries


def _entry_to_dict(e, source: str) -> Dict:
    # mlblack: e.contract is a dict with context_requires etc.
    # nsgablack: e.context_requires etc. are direct attributes
    if hasattr(e, "contract") and isinstance(e.contract, dict):
        requires = e.contract.get("context_requires", ())
        provides = e.contract.get("context_provides", ())
    else:
        requires = getattr(e, "context_requires", ()) or ()
        provides = getattr(e, "context_provides", ()) or ()
    return {
        "key": e.key,
        "title": e.title or e.key,
        "kind": e.kind,
        "summary": e.summary or "",
        "tags": list(e.tags) if e.tags else [],
        "import_path": e.import_path or "",
        "requires": list(requires),
        "provides": list(provides),
        "source": source,
    }


def search_entries(query: str, kind: Optional[str] = None, limit: int = 8) -> List[Dict]:
    """关键词搜索 catalog（含中英文术语映射）"""
    entries = _load_catalog()

    CN_MAP = {
        "多目标": "multi objective nsga nsga2 nsga3 moea spea pareto",
        "符号回归": "symbolic regression fixed expression basis",
        "梯度下降": "gradient descent backprop",
        "分类": "classification softmax logistic",
        "回归": "regression",
        "时序": "temporal time series neural",
        "约束": "constraint bias penalty",
        "图": "graph neural net gnn tsp vrp connectivity",
        "树": "tree ensemble xgboost gbdt",
        "神经网络": "neural network mlp backprop",
        "集成": "ensemble tree forest boosting",
        "降维": "dimensionality reduction tsne pca",
        "聚类": "cluster gmm",
        "搜索": "search de sa vns nsga random pattern",
        "旅行商": "tsp travelling hamiltonian",
        "车辆路径": "vrp routing",
        "调度": "scheduling",
        "因果": "causal graph sparsity",
        "异常检测": "anomaly outlier constraint",
        "模拟退火": "sa simulated annealing",
        "邻域搜索": "vns variable neighborhood",
        "差分进化": "de differential evolution",
        "禁忌搜索": "tabu tabu_search",
        "贝叶斯": "bayesian gaussian",
    }
    q = query.lower()
    english_q = q
    for cn, en in CN_MAP.items():
        if cn in q:
            english_q += " " + en

    scored = []
    for e in entries:
        if kind and e["kind"] != kind:
            continue
        text = f"{e['key']} {e['title']} {e['kind']} {e['summary']} {' '.join(e['tags'])}".lower()
        s = sum(1 for w in english_q.split() if w in text)
        if s > 0:
            scored.append((s, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:limit]]


def get_kinds() -> List[str]:
    return sorted(set(e["kind"] for e in _load_catalog()))
