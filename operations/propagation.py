import networkx as nx
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from typing import Dict

def _extract_run_logs(run_row):
    exec_log = run_row.get("execution_logs")
    if isinstance(exec_log, dict):
        return exec_log.get("logs", [])
    if isinstance(exec_log, list):
        return exec_log
    return []

def _safe_json(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value

def _collect_node_events(run_name, logs):
    node_starts = defaultdict(list)
    node_ends = defaultdict(list)
    edge_events = []

    for log in logs or []:
        if not isinstance(log, dict):
            continue

        event_type = log.get("event_type")
        node_id = log.get("node_id")
        details = log.get("details") or {}

        if event_type == "NODE_START" and node_id:
            node_starts[node_id].append(
                {
                    "timestamp": log.get("timestamp"),
                    "inputs": details.get("inputs", []),
                    "node_type": details.get("node_type"),
                    "input_count": details.get("input_count"),
                    "predecessors": details.get("predecessors", []),
                    "successors": details.get("successors", []),
                }
            )
        elif event_type == "NODE_END" and node_id:
            node_ends[node_id].append(
                {
                    "timestamp": log.get("timestamp"),
                    "output": details.get("output"),
                    "output_size": details.get("output_size"),
                    "output_count": details.get("output_count"),
                    "output_role": details.get("output_role"),
                    "output_source": details.get("output_source"),
                }
            )
        elif event_type == "EDGE_PROCESS" and node_id:
            edge_events.append(
                {
                    "run_name": run_name,
                    "source_node_id": node_id,
                    "target_node_id": details.get("to_node"),
                    "timestamp": log.get("timestamp"),
                    "condition": details.get("condition"),
                    "condition_label": details.get("condition_label"),
                    "condition_type": details.get("condition_type"),
                    "carry_data": details.get("carry_data"),
                    "keep_message": details.get("keep_message"),
                    "clear_context": details.get("clear_context"),
                    "clear_kept_context": details.get("clear_kept_context"),
                    "process": details.get("process"),
                    "process_type": details.get("process_type"),
                }
            )

    return node_starts, node_ends, edge_events

def _pick_first(items, key):
    if not items:
        return None
    return items[0].get(key)

def _build_node_relation_rows(run_name, node_starts, node_ends, edge_events):
    rows = []
    for edge in edge_events:
        source_node_id = edge["source_node_id"]
        target_node_id = edge["target_node_id"]

        source_start = _pick_first(node_starts.get(source_node_id, []), "timestamp")
        source_end = _pick_first(node_ends.get(source_node_id, []), "timestamp")
        source_output = _pick_first(node_ends.get(source_node_id, []), "output")

        target_start_record = _pick_first(node_starts.get(target_node_id, []), "timestamp")
        target_inputs = _pick_first(node_starts.get(target_node_id, []), "inputs")

        rows.append(
            {
                "run_name": run_name,
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
                "source_start_timestamp": source_start,
                "source_end_timestamp": source_end,
                "target_start_timestamp": target_start_record,
                "source_output": _safe_json(source_output),
                "target_input": _safe_json(target_inputs),
                "edge_condition": edge["condition"],
                "condition_label": edge["condition_label"],
                "condition_type": edge["condition_type"],
                "carry_data": edge["carry_data"],
                "keep_message": edge["keep_message"],
                "clear_context": edge["clear_context"],
                "clear_kept_context": edge["clear_kept_context"],
                "process": _safe_json(edge["process"]),
                "process_type": edge["process_type"],
            }
        )
    return rows

def build_hallucination_node_relations(warehouse_dataframe):
    hallucinated_rows = []
    relation_rows = []

    for _, run_row in warehouse_dataframe.iterrows():
        run_name = run_row.get("run_name")
        logs = _extract_run_logs(run_row)
        node_starts, node_ends, edge_events = _collect_node_events(run_name, logs)

        relation_rows.extend(_build_node_relation_rows(run_name, node_starts, node_ends, edge_events))

        qwen_results = run_row.get("qwen_results") or []
        for result in qwen_results:
            if not isinstance(result, dict):
                continue
            score = result.get("score") or (result.get("qwen") or {}).get("SCORE")
            if score != "HALLUCINATION DETECTED":
                continue

            qwen_payload = result.get("qwen") or {}
            hallucinated_rows.append(
                {
                    "run_name": run_name,
                    "node_id": result.get("node_id"),
                    "qwen_score": score,
                    "qwen_explanation": qwen_payload.get("EXPLANATION"),
                    "timestamp": result.get("timestamp"),
                }
            )

    hallucinated_nodes_df = pd.DataFrame(hallucinated_rows)
    node_relations_df = pd.DataFrame(relation_rows)

    if not hallucinated_nodes_df.empty:
        hallucinated_nodes_df["timestamp"] = pd.to_datetime(hallucinated_nodes_df["timestamp"], errors="coerce")

    if not node_relations_df.empty:
        node_relations_df["source_start_timestamp"] = pd.to_datetime(node_relations_df["source_start_timestamp"], errors="coerce")
        node_relations_df["source_end_timestamp"] = pd.to_datetime(node_relations_df["source_end_timestamp"], errors="coerce")
        node_relations_df["target_start_timestamp"] = pd.to_datetime(node_relations_df["target_start_timestamp"], errors="coerce")

    return hallucinated_nodes_df, node_relations_df

def identify_critical_scenario(graphs_dict):
    scenario_counts = {}

    for name, data in graphs_dict.items():
        name_parts = str(name).rsplit('_', 1)
        scenario = name_parts[0] if len(name_parts) > 1 else name

        current_graph = data['graph']
        hallucinations = sum(1 for n, attrs in current_graph.nodes(data=True) if attrs.get('hallucinated', False))

        if scenario not in scenario_counts:
            scenario_counts[scenario] = 0
        scenario_counts[scenario] += hallucinations

    critical_scenario = max(scenario_counts, key=scenario_counts.get)
    return critical_scenario

def compute_node_metrics_for_all_runs(graphs, hallucinated_nodes_df):
    rows = []
    for run_name, run_data in graphs.items():
        G = run_data.get("graph")
        if G is None or G.number_of_nodes() == 0:
            continue
        h_nodes = set(hallucinated_nodes_df[hallucinated_nodes_df["run_name"] == run_name]["node_id"].dropna().tolist())
        betweenness = nx.betweenness_centrality(G)
        try:
            pagerank = nx.pagerank(G)
        except Exception:
            pagerank = {n: 0.0 for n in G.nodes()}
        undirected = G.to_undirected() if G.number_of_nodes() > 0 else G
        clustering = nx.clustering(undirected) if hasattr(undirected, "nodes") else {}
        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())
        dist_map = {n: np.inf for n in G.nodes()}
        for s in h_nodes:
            if s not in G:
                continue
            lengths = nx.single_source_shortest_path_length(G, s)
            for n, d in lengths.items():
                if d < dist_map.get(n, np.inf):
                    dist_map[n] = d
        for n in G.nodes():
            preds = set(G.predecessors(n)) if G.is_directed() else set(G.neighbors(n))
            succs = set(G.successors(n)) if G.is_directed() else set()
            neighbors = preds | succs
            neighbor_h_rate = 0.0
            if neighbors:
                neighbor_h_rate = sum(1 for nb in neighbors if nb in h_nodes) / len(neighbors)
            rows.append({
                "run_name": run_name,
                "node_id": n,
                "is_hallucinated": n in h_nodes,
                "in_degree": float(in_deg.get(n, 0)),
                "out_degree": float(out_deg.get(n, 0)),
                "degree_total": float(in_deg.get(n, 0) + out_deg.get(n, 0)),
                "betweenness": float(betweenness.get(n, 0.0)),
                "pagerank": float(pagerank.get(n, 0.0)),
                "clustering": float(clustering.get(n, 0.0)),
                "neighbor_hallucinated_rate": float(neighbor_h_rate),
                "distance_to_nearest_hallucinated": (float(dist_map.get(n, np.inf)) if np.isfinite(dist_map.get(n, np.inf)) else np.nan)
            })
    return pd.DataFrame(rows)

def aggregate_node_metrics(nodes_df):
    agg = nodes_df.groupby("node_id").agg(
        hallucinations=("is_hallucinated", "sum"),
        appearances=("run_name", "nunique"),
        hallucination_rate=("is_hallucinated", "mean"),
        mean_in_degree=("in_degree", "mean"),
        mean_out_degree=("out_degree", "mean"),
        mean_degree_total=("degree_total", "mean"),
        mean_betweenness=("betweenness", "mean"),
        mean_pagerank=("pagerank", "mean"),
        mean_clustering=("clustering", "mean"),
        mean_neighbor_hallucinated_rate=("neighbor_hallucinated_rate", "mean"),
        mean_distance_to_hallucinated=("distance_to_nearest_hallucinated", "mean")
    ).reset_index()
    return agg

def compute_feature_correlations(agg_df, method="spearman"):
    numeric_cols = [
        "mean_in_degree", "mean_out_degree", "mean_degree_total", "mean_betweenness",
        "mean_pagerank", "mean_clustering", "mean_neighbor_hallucinated_rate", "mean_distance_to_hallucinated"
    ]
    present = [c for c in numeric_cols if c in agg_df.columns]
    corr = agg_df[["hallucination_rate"] + present].corr(method=method)["hallucination_rate"].drop("hallucination_rate")
    return corr

def score_and_explain_nodes(agg_df, correlations):
    features = correlations.index.tolist()
    abs_corr = correlations.abs()
    if abs_corr.sum() == 0:
        weights = pd.Series(1.0 / len(features), index=features)
    else:
        weights = abs_corr / abs_corr.sum()
    means = agg_df[features].mean()
    stds = agg_df[features].std().replace(0, np.nan)
    z = (agg_df[features] - means) / stds
    z = z.fillna(0)
    signed_weights = correlations.fillna(0)
    contribs = z.multiply(signed_weights, axis=1)
    agg_df = agg_df.copy()
    agg_df["propensity_score"] = contribs.sum(axis=1)
    top_features_per_node = []
    for idx, row in contribs.iterrows():
        feat_contrib = row.to_dict()
        sorted_feats = sorted(feat_contrib.items(), key=lambda x: -abs(x[1]))
        top_features_per_node.append([{"feature": f, "contribution": float(v)} for f, v in sorted_feats[:3]])
    agg_df["top_feature_contributions"] = top_features_per_node
    ranked = agg_df.sort_values(["hallucination_rate", "propensity_score"], ascending=[False, False]).reset_index(drop=True)
    return ranked

def analyze_node_proneness(graphs, hallucinated_nodes_df, top_n=20):
    nodes_df = compute_node_metrics_for_all_runs(graphs, hallucinated_nodes_df)
    if nodes_df.empty:
        return {"error": "No node data generated."}
    agg = aggregate_node_metrics(nodes_df)
    correlations = compute_feature_correlations(agg)
    ranked = score_and_explain_nodes(agg, correlations)
    top = ranked.head(top_n)
    return {
        "nodes_df": nodes_df,
        "aggregated_df": agg,
        "correlations": correlations,
        "ranked_nodes": ranked,
        "top_nodes": top
    }

def extract_global_feature_influence(top_nodes_df):
    feature_impact = defaultdict(float)
    feature_frequency = defaultdict(int)

    for contributions in top_nodes_df["top_feature_contributions"]:
        for item in contributions:
            feat = item["feature"]
            val = item["contribution"]
            
            feature_impact[feat] += abs(val)
            feature_frequency[feat] += 1

    summary = []
    for feat, impact in feature_impact.items():
        summary.append({
            "feature": feat,
            "total_absolute_impact": impact,
            "frequency_in_top": feature_frequency[feat]
        })
        
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values(by="total_absolute_impact", ascending=False).reset_index(drop=True)
    
    return summary_df

def summarize_contagion_patterns(downstream_metrics_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Summarize reproducible contagion patterns from downstream node-level metrics.

    Returns a dict with:
    - contagion_df: normalized and filtered downstream rows used for the analysis
    - node_contagion_df: per-node aggregated view across sources within each run
    - convergent_nodes_df: nodes with at least two hallucinated predecessors
    - source_pattern_df: per-source dominant pattern classification
    - run_pattern_summary_df: per-run counts and rates for convergent/divergent/mixed sources
    - distance_pattern_df: per-distance summary of convergent signals
    - global_pattern_summary_df: global distribution of dominant patterns
    """
    empty_result = {
        "contagion_df": pd.DataFrame(),
        "node_contagion_df": pd.DataFrame(),
        "convergent_nodes_df": pd.DataFrame(),
        "source_pattern_df": pd.DataFrame(),
        "run_pattern_summary_df": pd.DataFrame(),
        "distance_pattern_df": pd.DataFrame(),
        "global_pattern_summary_df": pd.DataFrame(),
    }

    if downstream_metrics_df.empty:
        return empty_result

    contagion_df = downstream_metrics_df.copy()
    required_columns = ["distance", "run_name", "hallucinated_source_id", "downstream_node_id"]
    contagion_df = contagion_df.dropna(subset=[column for column in required_columns if column in contagion_df.columns]).copy()
    if contagion_df.empty:
        return empty_result

    contagion_df["distance"] = pd.to_numeric(contagion_df["distance"], errors="coerce")
    contagion_df = contagion_df.dropna(subset=["distance"]).copy()
    if contagion_df.empty:
        return empty_result

    contagion_df["distance"] = contagion_df["distance"].astype(int)
    contagion_df["hallucinated_flag"] = contagion_df["hallucination_status"].eq("HALLUCINATION DETECTED")
    contagion_df["convergent_flag"] = pd.to_numeric(contagion_df["hallucinated_predecessors_count"], errors="coerce").fillna(0).astype(int) >= 2
    contagion_df["direct_child_flag"] = contagion_df["distance"].eq(1)

    node_contagion_df = (
        contagion_df.groupby(["run_name", "downstream_node_id"], as_index=False)
        .agg(
            max_distance=("distance", "max"),
            hallucinated_predecessors_count=("hallucinated_predecessors_count", "max"),
            hallucinated_flag=("hallucinated_flag", "max"),
            source_support=("hallucinated_source_id", "nunique"),
        )
    )
    convergent_nodes_df = node_contagion_df[node_contagion_df["hallucinated_predecessors_count"].fillna(0).astype(int) >= 2].copy()

    source_pattern_df = (
        contagion_df.groupby(["run_name", "hallucinated_source_id"], as_index=False)
        .agg(
            reachable_nodes=("downstream_node_id", "nunique"),
            direct_children=("direct_child_flag", "sum"),
            convergent_support=("convergent_flag", "sum"),
            hallucinated_descendants=("hallucinated_flag", "sum"),
            max_distance=("distance", "max"),
        )
    )
    source_pattern_df["dominant_pattern"] = np.select(
        [
            source_pattern_df["convergent_support"] > source_pattern_df["direct_children"],
            source_pattern_df["direct_children"] > source_pattern_df["convergent_support"],
        ],
        ["convergent", "divergent"],
        default="mixed",
    )

    run_pattern_rows = []
    for run_name, run_df in source_pattern_df.groupby("run_name"):
        total_sources = int(len(run_df))
        run_pattern_rows.append(
            {
                "run_name": run_name,
                "total_sources": total_sources,
                "convergent_sources": int((run_df["dominant_pattern"] == "convergent").sum()),
                "divergent_sources": int((run_df["dominant_pattern"] == "divergent").sum()),
                "mixed_sources": int((run_df["dominant_pattern"] == "mixed").sum()),
                "convergence_rate": float((run_df["dominant_pattern"] == "convergent").mean()) if total_sources else 0.0,
                "divergence_rate": float((run_df["dominant_pattern"] == "divergent").mean()) if total_sources else 0.0,
                "mean_reachable_nodes": float(run_df["reachable_nodes"].mean()) if total_sources else 0.0,
                "mean_direct_children": float(run_df["direct_children"].mean()) if total_sources else 0.0,
                "mean_convergent_support": float(run_df["convergent_support"].mean()) if total_sources else 0.0,
            }
        )
    run_pattern_summary_df = pd.DataFrame(run_pattern_rows).sort_values("run_name").reset_index(drop=True)

    distance_pattern_rows = []
    if not node_contagion_df.empty:
        unique_nodes_df = node_contagion_df.drop_duplicates(subset=["run_name", "downstream_node_id"]).copy()
        for dist, level_df in unique_nodes_df.groupby("max_distance"):
            distance_pattern_rows.append(
                {
                    "distance": int(dist),
                    "nodes_count": int(level_df["downstream_node_id"].nunique()),
                    "hallucinated_nodes": int(level_df["hallucinated_flag"].sum()),
                    "convergent_nodes": int((pd.to_numeric(level_df["hallucinated_predecessors_count"], errors="coerce").fillna(0).astype(int) >= 2).sum()),
                    "mean_hallucinated_predecessors": float(pd.to_numeric(level_df["hallucinated_predecessors_count"], errors="coerce").fillna(0).mean()),
                }
            )
    distance_pattern_df = pd.DataFrame(distance_pattern_rows).sort_values("distance").reset_index(drop=True) if distance_pattern_rows else pd.DataFrame()

    global_pattern_summary_df = (
        source_pattern_df["dominant_pattern"].value_counts().rename_axis("dominant_pattern").reset_index(name="sources_count")
    )

    return {
        "contagion_df": contagion_df,
        "node_contagion_df": node_contagion_df,
        "convergent_nodes_df": convergent_nodes_df,
        "source_pattern_df": source_pattern_df,
        "run_pattern_summary_df": run_pattern_summary_df,
        "distance_pattern_df": distance_pattern_df,
        "global_pattern_summary_df": global_pattern_summary_df,
    }
