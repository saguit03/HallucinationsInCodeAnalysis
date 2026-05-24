from .graphs import build_hallucination_graph_from_run, build_all_hallucination_graphs, get_all_graph_metrics, get_info_directed_graph
from .latency import correlate_latency_with_node_propensity, compute_edge_latencies, summarize_edge_latencies, compute_distance_latency_profile
from .metrics_extraction import extract_downstream_metrics, aggregate_by_distance, compute_reachability, evaluate_shapiro_normality, calculate_homogeneous_correlations
from .propagation import identify_critical_scenario, analyze_node_proneness, aggregate_node_metrics, build_hallucination_node_relations, compute_node_metrics_for_all_runs, summarize_contagion_patterns, extract_global_feature_influence
from .traces import load_warehouse_run_jsons, get_logs_from_warehouse
from .visualization import plot_contagion_support_distribution, visualize_log_data, plot_aggregation_results, plot_latency_distribution, plot_betweenness_latency_correlation, visualize_propagation_patient_zero, plot_target_heatmap, plot_global_correlation_matrix, plot_optimized_distributions

from .globals import SEED, EXECUTION_FILE, QWEN_FILE, RESULTS_PATH, WAREHOUSE_PATH, ALPHA_VALUE, VISUALIZATION_DIR, DPI
from pathlib import Path

Path(RESULTS_PATH).mkdir(exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    "SEED",
    "EXECUTION_FILE",
    "QWEN_FILE",
    "RESULTS_PATH",
    "WAREHOUSE_PATH",
    "ALPHA_VALUE",
    "VISUALIZATION_DIR",
    "DPI",
    "compute_reachability",
    "summarize_contagion_patterns",
    "extract_downstream_metrics",
    "aggregate_by_distance",
    "build_hallucination_graph_from_run",
    "build_all_hallucination_graphs",
    "identify_critical_scenario",
    "visualize_propagation_patient_zero",
    "analyze_node_proneness",
    "compute_edge_latencies",
    "summarize_edge_latencies",
    "get_all_graph_metrics",
    "get_info_directed_graph",
    "evaluate_shapiro_normality",
    "calculate_homogeneous_correlations",
    "plot_target_heatmap",
    "plot_global_correlation_matrix",
    "plot_optimized_distributions",
    "build_hallucination_node_relations",
    "plot_contagion_support_distribution",
    "compute_distance_latency_profile",
    "correlate_latency_with_node_propensity",
    "load_warehouse_run_jsons",
    "get_logs_from_warehouse",
    "compute_node_metrics_for_all_runs",
    "aggregate_node_metrics",
    "visualize_log_data",
    "plot_aggregation_results",
    "plot_latency_distribution",
    "plot_betweenness_latency_correlation",
    "extract_global_feature_influence"
]
