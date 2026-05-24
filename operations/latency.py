import pandas as pd
import numpy as np
from scipy import stats

def compute_distance_latency_profile(downstream_df, edge_latency_df):
    merged = downstream_df[["run_name", "hallucinated_source_id", "downstream_node_id", "distance"]].merge(
        edge_latency_df[["run_name", "source_node_id", "target_node_id", "edge_latency_ms"]],
        left_on=["run_name", "hallucinated_source_id", "downstream_node_id"],
        right_on=["run_name", "source_node_id", "target_node_id"],
        how="left"
    )
    
    merged = merged.drop(columns=["source_node_id", "target_node_id"], errors="ignore")
    merged["distance"] = pd.to_numeric(merged["distance"], errors="coerce")
    merged["edge_latency_ms"] = pd.to_numeric(merged["edge_latency_ms"], errors="coerce")
    merged = merged[pd.notna(merged["distance"]) & pd.notna(merged["edge_latency_ms"])]
    
    return merged

def correlate_latency_with_node_propensity(merged_latency, result, alpha=0.05):
    propensity_df = result["aggregated_df"][["node_id", "hallucination_rate", "mean_betweenness", "mean_in_degree", "mean_pagerank"]].copy()
    propensity_df.columns = ["node_id", "halluc_rate", "betweenness", "in_degree", "pagerank"]
    propensity_df["node_id"] = propensity_df["node_id"].astype(str)
    
    merged_latency_copy = merged_latency.copy()
    merged_latency_copy["hallucinated_source_id"] = merged_latency_copy["hallucinated_source_id"].astype(str)
    corr_data = merged_latency_copy.merge(
        propensity_df,
        left_on="hallucinated_source_id",
        right_on="node_id",
        how="left"
    )
    corr_data = corr_data[pd.notna(corr_data["edge_latency_ms"])].copy()
    topological_metrics = ["halluc_rate", "betweenness", "in_degree", "pagerank"]
    target_column = "edge_latency_ms"
    target_data = corr_data[target_column].dropna()
    if len(target_data) >= 8:
        _, target_p_value = stats.normaltest(target_data)
        all_normal = bool(target_p_value > alpha)
    else:
        all_normal = False
    if all_normal:
        for metric in topological_metrics:
            metric_data = corr_data[metric].dropna()
            if len(metric_data) >= 8:
                _, metric_p_value = stats.normaltest(metric_data)
                if metric_p_value <= alpha:
                    all_normal = False
                    break
            else:
                all_normal = False
                break
    method_used = 'pearson' if all_normal else 'spearman'
    
    results = {}
    for metric in topological_metrics:
        clean_df = corr_data[[metric, target_column]].dropna()
        if len(clean_df) > 1:
            if method_used == 'spearman':
                coef, corr_p_value = stats.spearmanr(clean_df[metric], clean_df[target_column])
            else:
                coef, corr_p_value = stats.pearsonr(clean_df[metric], clean_df[target_column])
        else:
            coef, corr_p_value = np.nan, np.nan
        results[f"latency_vs_{metric}"] = {
            'coeficiente': coef,
            'p_valor': corr_p_value
        }
    return pd.DataFrame(results).T, method_used, corr_data
import pandas as pd

def compute_edge_latencies(node_relations_df):
    edge_latency_df = node_relations_df.copy()
    edge_latency_df["source_end_timestamp"] = pd.to_datetime(edge_latency_df["source_end_timestamp"], errors="coerce")
    edge_latency_df["target_start_timestamp"] = pd.to_datetime(edge_latency_df["target_start_timestamp"], errors="coerce")
    edge_latency_df["edge_latency_ms"] = (
        (edge_latency_df["target_start_timestamp"] - edge_latency_df["source_end_timestamp"]).dt.total_seconds() * 1000
    )
    edge_latency_df.loc[
        (edge_latency_df["edge_latency_ms"] < 0) & (edge_latency_df["edge_latency_ms"] >= -1000), 
        "edge_latency_ms"
    ] = 0
    edge_latency_df = edge_latency_df[edge_latency_df["edge_latency_ms"] >= 0].copy()
    edge_latency_df = edge_latency_df[
        ["run_name", "source_node_id", "target_node_id", "source_end_timestamp", 
         "target_start_timestamp", "edge_latency_ms", "process_type"]
    ].copy()
    return edge_latency_df

def summarize_edge_latencies(edge_latency_df):
    valid = edge_latency_df[pd.notna(edge_latency_df["edge_latency_ms"])].copy()
    
    if valid.empty:
        return pd.DataFrame()
    
    summary_rows = []
    
    global_stats = {
        "metric": "global",
        "mean_latency_ms": float(valid["edge_latency_ms"].mean()),
        "median_latency_ms": float(valid["edge_latency_ms"].median()),
        "std_latency_ms": float(valid["edge_latency_ms"].std()),
        "min_latency_ms": float(valid["edge_latency_ms"].min()),
        "max_latency_ms": float(valid["edge_latency_ms"].max()),
        "p25_latency_ms": float(valid["edge_latency_ms"].quantile(0.25)),
        "p75_latency_ms": float(valid["edge_latency_ms"].quantile(0.75)),
        "count": len(valid)
    }
    summary_rows.append(global_stats)
    
    for run_name in valid["run_name"].unique():
        run_valid = valid[valid["run_name"] == run_name]
        summary_rows.append({
            "metric": f"run_{run_name}",
            "mean_latency_ms": float(run_valid["edge_latency_ms"].mean()),
            "median_latency_ms": float(run_valid["edge_latency_ms"].median()),
            "std_latency_ms": float(run_valid["edge_latency_ms"].std()),
            "min_latency_ms": float(run_valid["edge_latency_ms"].min()),
            "max_latency_ms": float(run_valid["edge_latency_ms"].max()),
            "p25_latency_ms": float(run_valid["edge_latency_ms"].quantile(0.25)),
            "p75_latency_ms": float(run_valid["edge_latency_ms"].quantile(0.75)),
            "count": len(run_valid)
        })
    
    for process_type in valid["process_type"].dropna().unique():
        proc_valid = valid[valid["process_type"] == process_type]
        summary_rows.append({
            "metric": f"process_{process_type}",
            "mean_latency_ms": float(proc_valid["edge_latency_ms"].mean()),
            "median_latency_ms": float(proc_valid["edge_latency_ms"].median()),
            "std_latency_ms": float(proc_valid["edge_latency_ms"].std()),
            "min_latency_ms": float(proc_valid["edge_latency_ms"].min()),
            "max_latency_ms": float(proc_valid["edge_latency_ms"].max()),
            "p25_latency_ms": float(proc_valid["edge_latency_ms"].quantile(0.25)),
            "p75_latency_ms": float(proc_valid["edge_latency_ms"].quantile(0.75)),
            "count": len(proc_valid)
        })
    
    return pd.DataFrame(summary_rows)

