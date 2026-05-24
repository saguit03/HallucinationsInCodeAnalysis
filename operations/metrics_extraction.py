import pandas as pd
import networkx as nx
from typing import Dict, Any, Optional, List, Tuple
from collections import deque, defaultdict
from scipy import stats
from operations.globals import ALPHA_VALUE

def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default

def _bfs_distances(G: nx.DiGraph, source) -> Dict[Any, int]:
    distances = {}
    q = deque()
    q.append((source, 0))
    visited = {source}
    while q:
        node, d = q.popleft()
        for nbr in G.successors(node):
            if nbr not in visited:
                visited.add(nbr)
                distances[nbr] = d + 1
                q.append((nbr, d + 1))
    return distances


def _enumerate_simple_paths_from_source(G: nx.DiGraph, source) -> Dict[Any, List[List[Any]]]:
    paths_per_target = defaultdict(list)

    def dfs(node, path, visited):
        paths_per_target[node].append(list(path))
        for nbr in G.successors(node):
            if nbr in visited:
                continue
            visited.add(nbr)
            path.append(nbr)
            dfs(nbr, path, visited)
            path.pop()
            visited.remove(nbr)

    dfs(source, [source], {source})
    return paths_per_target


def compute_reachability(
    G: nx.DiGraph,
    hallucinated_nodes: List[Any],
    run_name: str = "run",
) -> Tuple[Dict[Any, Dict], pd.DataFrame, Dict[str, Any]]:
    """
    For each hallucinated node compute downstream reachability until graph termination.

    Returns:
    - reachability_map: {hallucinated_node: {distance: [nodes], paths: {"all": [...], "per_target": {...}}, path_length_distribution: {...}}}
    - cascade_depths_df: DataFrame with columns [run_name, hallucinated_node_id, cascade_depth, affected_nodes_count, path_count]
    - stats: dict with keys total_paths, avg_cascade_depth, max_cascade_depth
    """
    reachability_map = {}
    rows = []
    total_paths = 0
    depths = []

    for src in hallucinated_nodes:
        if src not in G:
            reachability_map[src] = {"distance": {}, "paths": {"all": [], "per_target": {}}, "path_length_distribution": {}}
            rows.append((run_name, src, 0, 0, 0))
            depths.append(0)
            continue

        distances = _bfs_distances(G, src)  # node -> distance (int)
        # bucket by distance
        distance_buckets = defaultdict(list)
        for node, d in distances.items():
            distance_buckets[d].append(node)

        # enumerate all simple paths starting at src (records paths to all intermediate and terminal nodes)
        per_target_paths = _enumerate_simple_paths_from_source(G, src)

        # flatten all paths (exclude trivial path [src] from "all")
        all_paths = []
        for target, paths in per_target_paths.items():
            for p in paths:
                if len(p) > 1:
                    all_paths.append(p)

        # compute path length distribution per target
        path_length_distribution = {}
        for target, paths in per_target_paths.items():
            lengths = defaultdict(int)
            for p in paths:
                if len(p) <= 1:
                    continue
                lengths[len(p) - 1] += 1
            if lengths:
                path_length_distribution[target] = dict(lengths)

        reachability_map[src] = {
            "distance": {d: distance_buckets[d] for d in sorted(distance_buckets.keys())},
            "paths": {"all": all_paths, "per_target": dict(per_target_paths)},
            "path_length_distribution": path_length_distribution,
        }

        cascade_depth = max(distances.values()) if distances else 0
        affected_nodes_count = len(distances)
        path_count = len(all_paths)
        total_paths += path_count
        depths.append(cascade_depth)
        rows.append((run_name, src, int(cascade_depth), int(affected_nodes_count), int(path_count)))

    cascade_depths_df = pd.DataFrame(rows, columns=["run_name", "hallucinated_node_id", "cascade_depth", "affected_nodes_count", "path_count"])

    stats = {
        "total_paths": int(total_paths),
        "avg_cascade_depth": float(sum(depths) / len(depths)) if depths else 0.0,
        "max_cascade_depth": int(max(depths)) if depths else 0,
    }

    return reachability_map, cascade_depths_df, stats

def extract_downstream_metrics(
    G: nx.DiGraph,
    reachability_map: Dict[Any, Dict],
    run_name: str,
    node_relations_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build downstream_metrics_df for all hallucinated sources present in reachability_map.

    Returns a DataFrame with columns:
    [run_name, hallucinated_source_id, downstream_node_id, distance,
     hallucination_status, in_degree, out_degree, betweenness, pagerank,
     hallucinated_predecessors_count, execution_timing]

    `reachability_map` is expected in the shape returned by `compute_reachability`.
    If `node_relations_df` is provided it should contain `run_name` and `target_node_id`
    and time columns (e.g., `target_start_timestamp`) to populate execution_timing.
    """
    rows = []
    clustering_lookup = {}
    if G.number_of_nodes() > 0:
        try:
            clustering_lookup = nx.clustering(G.to_undirected())
        except Exception:
            clustering_lookup = {}

    # prepare timing lookup if node_relations_df provided
    timing_lookup = {}
    if node_relations_df is not None and not node_relations_df.empty:
        df = node_relations_df.copy()
        if "run_name" in df.columns:
            df = df[df["run_name"] == run_name]
        # choose earliest target_start_timestamp per target
        if "target_node_id" in df.columns and "target_start_timestamp" in df.columns:
            grp = df.groupby("target_node_id")["target_start_timestamp"].min()
            timing_lookup = grp.to_dict()

    for src, data in reachability_map.items():
        # distances: {d: [nodes]}
        distances = data.get("distance", {})
        node_to_distance = {}
        for d, nodes in distances.items():
            for n in nodes:
                node_to_distance[n] = int(d)

        # for each reachable node record metrics
        for node, dist in node_to_distance.items():
            # node attrs from graph if present
            nattrs: Dict[str, Any] = dict(G.nodes[node]) if node in G else {}
            hallucinated = nattrs.get("hallucinated", False)
            in_deg = int(nattrs.get("in_degree", G.in_degree(node) if node in G else 0))
            out_deg = int(nattrs.get("out_degree", G.out_degree(node) if node in G else 0))
            betw = float(nattrs.get("betweenness", 0.0))
            pr = float(nattrs.get("pagerank", 0.0))
            clustering = float(clustering_lookup.get(node, 0.0))

            # count incoming hallucinated predecessors
            h_pred_count = 0
            if node in G:
                for p in G.predecessors(node):
                    if G.nodes.get(p, {}).get("hallucinated", False):
                        h_pred_count += 1

            exec_timing = timing_lookup.get(node)

            rows.append(
                {
                    "run_name": run_name,
                    "hallucinated_source_id": src,
                    "downstream_node_id": node,
                    "distance": _coerce_int(dist),
                    "hallucination_status": "HALLUCINATION DETECTED" if hallucinated else "NO HALLUCINATION",
                    "in_degree": in_deg,
                    "out_degree": out_deg,
                    "betweenness": betw,
                    "pagerank": pr,
                    "clustering": clustering,
                    "hallucinated_predecessors_count": int(h_pred_count),
                    "execution_timing": exec_timing,
                }
            )

    df = pd.DataFrame(rows)

    # ensure columns and dtypes
    expected_cols = [
        "run_name",
        "hallucinated_source_id",
        "downstream_node_id",
        "distance",
        "hallucination_status",
        "in_degree",
        "out_degree",
        "betweenness",
        "pagerank",
        "clustering",
        "hallucinated_predecessors_count",
        "execution_timing",
    ]

    for c in expected_cols:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[expected_cols]

    # cast numeric types where appropriate
    df["distance"] = df["distance"].astype("Int64")
    df["in_degree"] = df["in_degree"].astype("Int64")
    df["out_degree"] = df["out_degree"].astype("Int64")
    df["hallucinated_predecessors_count"] = df["hallucinated_predecessors_count"].astype("Int64")

    return df


def aggregate_by_distance(downstream_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate node-level metrics by distance level.

    Returns a DataFrame indexed by `distance` with aggregated columns:
    - nodes_count
    - hallucinated_count
    - hallucination_rate (0-1)
    - avg/median/max for degree, clustering, pagerank
    - avg_hallucinated_predecessors_count
    """
    if downstream_metrics_df.empty:
        return pd.DataFrame()

    df = downstream_metrics_df.copy()
    df_numeric = df.dropna(subset=["distance"]).copy()
    df_numeric["distance"] = df_numeric["distance"].astype(int)

    def safe_mean(s):
        try:
            return float(s.mean())
        except Exception:
            return None

    def safe_median(s):
        try:
            return float(s.median())
        except Exception:
            return None

    def safe_max(s):
        try:
            return float(s.max())
        except Exception:
            return None

    grouped = df_numeric.groupby("distance")
    rows = []
    for dist, g in grouped:
        nodes_count = int(g["downstream_node_id"].nunique())
        hallucinated_count = int((g["hallucination_status"] == "HALLUCINATION DETECTED").sum())
        hallucination_rate = hallucinated_count / nodes_count if nodes_count else 0.0
        total_degree = g["in_degree"].fillna(0) + g["out_degree"].fillna(0)
        clustering = g["clustering"] if "clustering" in g.columns else pd.Series(dtype=float)
        rows.append(
            {
                "distance": _coerce_int(dist),
                "nodes_count": nodes_count,
                "cascade_width": nodes_count,
                "hallucinated_count": hallucinated_count,
                "hallucination_rate": float(hallucination_rate),
                "avg_in_degree": safe_mean(g["in_degree"]),
                "median_in_degree": safe_median(g["in_degree"]),
                "max_in_degree": safe_max(g["in_degree"]),
                "avg_out_degree": safe_mean(g["out_degree"]),
                "median_out_degree": safe_median(g["out_degree"]),
                "max_out_degree": safe_max(g["out_degree"]),
                "avg_total_degree": safe_mean(total_degree),
                "median_total_degree": safe_median(total_degree),
                "max_total_degree": safe_max(total_degree),
                "avg_betweenness": safe_mean(g["betweenness"]),
                "median_betweenness": safe_median(g["betweenness"]),
                "max_betweenness": safe_max(g["betweenness"]),
                "avg_pagerank": safe_mean(g["pagerank"]),
                "median_pagerank": safe_median(g["pagerank"]),
                "max_pagerank": safe_max(g["pagerank"]),
                "avg_clustering": safe_mean(clustering),
                "median_clustering": safe_median(clustering),
                "max_clustering": safe_max(clustering),
                "avg_hallucinated_predecessors_count": safe_mean(g["hallucinated_predecessors_count"]),
            }
        )

    agg_df = pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)
    return agg_df


def evaluate_shapiro_normality(df, columns, alpha):
    results = {}
    for column in columns:
        clean_data = df[column].dropna()
        w_statistic, p_value = stats.shapiro(clean_data)
        results[column] = {
            'estadistico_w': w_statistic,
            'p_valor': p_value,
            'distribucion_normal': p_value > alpha
        }
    return pd.DataFrame(results).T

def calculate_homogeneous_correlations(df, topological_metrics, target_column, alpha=ALPHA_VALUE):
    target_data = df[target_column].dropna()
    _, target_p_value = stats.shapiro(target_data)
    all_normal = bool(target_p_value > alpha)

    if all_normal:
        for metric in topological_metrics:
            metric_data = df[metric].dropna()
            _, metric_p_value = stats.shapiro(metric_data)
            if metric_p_value <= alpha:
                all_normal = False
                break

    method_used = 'pearson' if all_normal else 'spearman'
    
    results = {}
    for metric in topological_metrics:
        clean_df = df[[metric, target_column]].dropna()
        
        if method_used == 'spearman':
            coef, corr_p_value = stats.spearmanr(clean_df[metric], clean_df[target_column])
        else:
            coef, corr_p_value = stats.pearsonr(clean_df[metric], clean_df[target_column])
            
        results[metric] = {
            'coeficiente': coef,
            'p_valor': corr_p_value
        }
        
    return pd.DataFrame(results).T, method_used
