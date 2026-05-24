from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import Any
from operations.globals import VISUALIZATION_DIR, DPI, FONTSIZE
AXIS_TICK_FONTSIZE = FONTSIZE * 0.7
BAR_WIDTH = 0.55

def _prepare_output_path(filename: str | None) -> Path | None:
    if not filename:
        return None
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    return VISUALIZATION_DIR / filename

def visualize_log_data(logs_df):
    level_counts = logs_df['level'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=level_counts.values, y=level_counts.index, ax=ax, palette='viridis', hue=level_counts.index)
    _add_horizontal_bar_labels(ax)
    ax.set_title('Proporción de trazas por nivel de log', fontsize=FONTSIZE)
    ax.set_xlabel('Cantidad de trazas', fontsize=FONTSIZE)
    ax.set_ylabel('Nivel', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=AXIS_TICK_FONTSIZE)
    plt.tight_layout()
    output_path = _prepare_output_path('proporcion_niveles_logs.png')
    if output_path is not None:
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.show()

    node_counts = logs_df['node_id'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=node_counts.values, y=node_counts.index, ax=ax, palette='magma', hue=node_counts.index)
    _add_horizontal_bar_labels(ax)
    ax.set_title('Proporción de trazas por nodo', fontsize=FONTSIZE)
    ax.set_xlabel('Cantidad de trazas', fontsize=FONTSIZE)
    ax.set_ylabel('Identificador de nodo', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=AXIS_TICK_FONTSIZE)
    plt.tight_layout()
    output_path = _prepare_output_path('proporcion_trazas_nodo.png')
    if output_path is not None:
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.show()

    event_counts = logs_df['event_type'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=event_counts.values, y=event_counts.index, ax=ax, palette='viridis', hue=event_counts.index, width=BAR_WIDTH)
    _add_horizontal_bar_labels(ax)
    ax.set_title('Proporción de trazas por tipo de evento', fontsize=FONTSIZE)
    ax.set_xlabel('Cantidad de trazas', fontsize=FONTSIZE)
    ax.set_ylabel('Tipo de evento', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=AXIS_TICK_FONTSIZE)
    plt.tight_layout()
    output_path = _prepare_output_path('proporcion_eventos.png')
    if output_path is not None:
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=logs_df['duration'].dropna(), ax=ax, log_scale=True, color='#4c72b0')
    ax.set_title('Distribución de la duración de eventos', fontsize=FONTSIZE)
    ax.set_xlabel('Duración en milisegundos', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=AXIS_TICK_FONTSIZE)
    plt.tight_layout()
    output_path = _prepare_output_path('distribucion_duracion.png')
    if output_path is not None:
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.show()


def _add_horizontal_bar_labels(ax):
    max_width = max((patch.get_width() for patch in ax.patches), default=0)
    for patch in ax.patches:
        width = patch.get_width()
        if width is None or width <= 0:
            continue

        y_pos = patch.get_y() + patch.get_height() / 2
        if max_width and width < max_width * 0.85:
            x_pos = width + max(max_width * 0.01, 0.5)
            ha = 'left'
            color = 'black'
        else:
            x_pos = width - max(width * 0.02, 0.5)
            ha = 'right'
            color = 'white'
        ax.text(
            x_pos,
            y_pos,
            f'{int(round(width))}',
            va='center',
            ha=ha,
            color=color,
            fontsize=AXIS_TICK_FONTSIZE,
            fontweight='bold',
        )

def plot_contagion_support_distribution(source_pattern_df: pd.DataFrame, save_filename: str | None = None):
    if source_pattern_df.empty:
        return None

    melted_df = source_pattern_df[["dominant_pattern", "direct_children", "convergent_support"]].copy()
    melted_df = melted_df.melt(id_vars="dominant_pattern", var_name="metric", value_name="value")
    if melted_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=melted_df,
        x="dominant_pattern",
        y="value",
        hue="metric",
        split=True,
        inner="quartile",
        palette={"direct_children": "#ff7f0e", "convergent_support": "#1f77b4"},
        ax=ax,
    )
    ax.set_title("Distribución del soporte de contagio por patrón", fontsize=FONTSIZE)
    ax.set_xlabel("Patrón dominante", fontsize=FONTSIZE)
    ax.set_ylabel("Valor", fontsize=FONTSIZE)
    ax.legend(title="Métrica", loc="best", fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=AXIS_TICK_FONTSIZE)
    plt.tight_layout()

    output_path = _prepare_output_path(save_filename)
    if output_path is not None:
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.show()

def plot_aggregation_results(agg_df: pd.DataFrame):
    df = agg_df.copy()
    
    df["distance"] = df["distance"].astype(str)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color_nodos = "#6baed6"
    sns.barplot(data=df, x="distance", y="nodes_count", color=color_nodos, ax=ax1)
    
    ax2 = ax1.twinx()
    
    color_alucinacion = "#d62728"
    sns.lineplot(data=df, x="distance", y="hallucination_rate", marker="o", color=color_alucinacion, linewidth=2, ax=ax2)
    
    ax1.set_xlabel("Distancia", fontsize=FONTSIZE)
    ax1.set_ylabel("Nodos (ancho de cascada)", color=color_nodos, fontsize=FONTSIZE)
    ax1.tick_params(axis="y", labelcolor=color_nodos, labelsize=AXIS_TICK_FONTSIZE)
    ax1.tick_params(axis="x", rotation=0, labelsize=AXIS_TICK_FONTSIZE)
    # ax1.bar_label(ax1.containers[0], fmt='%d', label_type='edge', color='#2b5c8f', fontweight='bold', padding=3)
    
    ax1.grid(True, axis='y', linestyle='--', color=color_nodos, alpha=0.6)
    ax2.grid(True, axis='y', linestyle='--', color=color_alucinacion, alpha=0.3)
    
    ax2.set_ylabel("Tasa de alucinación", color=color_alucinacion, fontsize=FONTSIZE)
    ax2.tick_params(axis="y", labelcolor=color_alucinacion, labelsize=AXIS_TICK_FONTSIZE)
    ax1.set_title("Evolución de la tasa de alucinación y ancho de cascada por distancia", fontsize=FONTSIZE)
    
    plt.tight_layout()
    output_path = _prepare_output_path("hallucination_rate_by_distance.png")
    if output_path is not None:
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.show()

def _find_column(df, keywords):
    for k in keywords:
        for c in df.columns:
            if k in c.lower():
                return c
    return None

def plot_latency_distribution(df, value_col=None, bins=60, kde=True, log_scale=False, figsize=(8,4)):
    if value_col is None:
        value_col = _find_column(df, ["latency","edge_latency","edge_latency_ms","latency_ms"])
    if value_col is None:
        raise ValueError("No se encontró columna de latencia")
    s = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if s.empty:
        raise ValueError("Columna de latencia vacía tras limpieza")
    plt.figure(figsize=figsize)
    sns.histplot(s, bins=bins, kde=kde)
    if log_scale:
        plt.yscale("log")
    plt.xlabel("Latencia (ms)", fontsize=AXIS_TICK_FONTSIZE)
    plt.title("Distribución de latencias", fontsize=AXIS_TICK_FONTSIZE)
    plt.tick_params(axis='both', labelsize=AXIS_TICK_FONTSIZE)
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR/"latency_distribution.png")
    plt.show()


def plot_betweenness_latency_correlation(corr_data):
    corr_data['betweenness_tier'] = pd.qcut(
        corr_data['betweenness'], 
        q=4, 
        labels=['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto'],
        duplicates='drop'
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.regplot(
        data=corr_data, 
        x='betweenness', 
        y='edge_latency_ms', 
        scatter_kws={'alpha': 0.4}, 
        line_kws={'color': 'darkred'},
        ax=axes[0]
    )
    axes[0].set_title('Latencia vs Betweenness Centrality', fontsize=FONTSIZE)
    axes[0].set_xlabel('Betweenness Centrality', fontsize=FONTSIZE)
    axes[0].set_ylabel('Latencia (ms)', fontsize=FONTSIZE)
    axes[0].tick_params(axis='both', labelsize=AXIS_TICK_FONTSIZE)

    sns.boxplot(
        data=corr_data, 
        x='betweenness_tier', 
        y='edge_latency_ms',
        log_scale=True,
        ax=axes[1]
    )
    axes[1].set_title('Distribución de Latencia por Nivel de Betweenness', fontsize=FONTSIZE)
    axes[1].set_xlabel('Nivel de Betweenness', fontsize=FONTSIZE)
    axes[1].set_ylabel('Latencia (ms)', fontsize=FONTSIZE)
    axes[1].tick_params(axis='both', labelsize=AXIS_TICK_FONTSIZE)

    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "betweenness_latency_correlation.png", dpi=DPI, bbox_inches='tight')
    plt.show()


def find_first_hallucinated(graph):
    for node, attrs in graph.nodes(data=True):
        if attrs.get('hallucinated', False):
            return node
    return None

def visualize_propagation_patient_zero(graphs_dict, scenario, columns=5, base_size=100, node_scale_factor=200, edge_scale_factor=1.4, figure_width=40, height_per_row=20):
    iterations = {name: data for name, data in graphs_dict.items() if scenario in str(name)}

    union_graph = nx.DiGraph()
    for data in iterations.values():
        union_graph.add_nodes_from(data['graph'].nodes())

    fixed_positions = nx.spring_layout(union_graph, seed=42)

    n_graphs = len(iterations)
    actual_columns = min(n_graphs, columns)
    rows = (n_graphs + actual_columns - 1) // actual_columns if actual_columns > 0 else 1
    
    adjusted_width = figure_width * (actual_columns / columns)

    fig, axes = plt.subplots(rows, actual_columns, figsize=(adjusted_width, height_per_row * rows))

    if isinstance(axes, np.ndarray):
        flat_axes = axes.flatten()
    else:
        flat_axes = [axes]

    for ax, (name, data) in zip(flat_axes, iterations.items()):
        G = data['graph']

        patient_zero = find_first_hallucinated(G)

        draw_graph = nx.DiGraph()
        for u, v in G.edges():
            if draw_graph.has_edge(u, v):
                draw_graph[u][v]['weight'] += 1
            else:
                draw_graph.add_edge(u, v, weight=1)

        for node in G.nodes():
            draw_graph.add_node(node, **G.nodes[node])

        node_color_map = []
        node_sizes = []

        for node in draw_graph.nodes():
            attrs = draw_graph.nodes[node]
            out_deg = G.out_degree(node)
            node_sizes.append(base_size + out_deg * node_scale_factor)

            if node == patient_zero:
                node_color_map.append('red')
            elif attrs.get('hallucinated', False):
                node_color_map.append('orange')
            else:
                node_color_map.append('#3498db')

        edge_color_map = []
        edge_widths = []

        for source, target, edge_attrs in draw_graph.edges(data=True):
            freq = edge_attrs['weight']
            edge_widths.append(freq * edge_scale_factor)

            if draw_graph.nodes[source].get('hallucinated', False):
                edge_color_map.append('#f39c12')
            else:
                edge_color_map.append('#bdc3c7')

        nx.draw(
            draw_graph,
            pos=fixed_positions,
            ax=ax,
            node_color=node_color_map,
            node_size=node_sizes,
            with_labels=True,
            font_size=12,
            edge_color=edge_color_map,
            width=edge_widths,
            arrows=True,
        )
        ax.set_title(name, fontsize=12)
        ax.tick_params(axis='both', labelsize=12)

    for i in range(n_graphs, len(flat_axes)):
        fig.delaxes(flat_axes[i])

    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / f"patient_zero.png", dpi=DPI, bbox_inches='tight')
    plt.show()

    return iterations


def plot_target_heatmap(correlation_df, coefficient_column, pvalue_column, significance_threshold):
    df_plot = correlation_df.copy()

    def _is_valid_scalar(value: Any):
        if not np.isscalar(value) or value is pd.NA:
            return False
        try:
            return not bool(np.isnan(value))
        except TypeError:
            return value is not None

    df_plot = df_plot[
        df_plot[coefficient_column].apply(_is_valid_scalar)
        & df_plot[pvalue_column].apply(_is_valid_scalar)
    ].copy()

    df_plot[coefficient_column] = pd.to_numeric(df_plot[coefficient_column], errors="coerce")
    df_plot[pvalue_column] = pd.to_numeric(df_plot[pvalue_column], errors="coerce")
    df_plot = df_plot.dropna(subset=[coefficient_column, pvalue_column])

    if df_plot.empty:
        print("No hay correlaciones numéricas válidas para graficar.")
        return

    sorted_df = df_plot.sort_values(by=coefficient_column, ascending=True)
    coefficients_matrix = sorted_df[[coefficient_column]].T

    annotations = np.empty((1, len(sorted_df)), dtype=object)
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        coef = row[coefficient_column]
        pval = row[pvalue_column]
        asterisk = "*" if pval < significance_threshold else ""
        annotations[0, i] = f"{coef:.2f}{asterisk}"

    figure, (cbar_axis, heatmap_axis) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(max(5, 0.6 * len(sorted_df)), 3.5),
        gridspec_kw={"height_ratios": [0.12, 0.88], "hspace": 0.5}
    )

    sns.heatmap(
        coefficients_matrix,
        annot=annotations,
        fmt="",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        cbar_ax=cbar_axis,
        cbar_kws={"label": "Coeficiente de correlación", "orientation": "horizontal"},
        ax=heatmap_axis
    )
    
    cbar_axis.xaxis.set_ticks_position('top')
    cbar_axis.xaxis.set_label_position('top')

    cbar_axis.set_title("Correlación con proporción de alucinaciones", pad=20, fontsize=AXIS_TICK_FONTSIZE, fontweight='bold')
    heatmap_axis.set_yticklabels([])
    cbar_axis.tick_params(axis='x', labelsize=AXIS_TICK_FONTSIZE)
    heatmap_axis.tick_params(axis='both', labelsize=AXIS_TICK_FONTSIZE)
    heatmap_axis.set_xticklabels(
        heatmap_axis.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
        fontsize=AXIS_TICK_FONTSIZE
    )
    
    plt.savefig(VISUALIZATION_DIR / "correlacion_mapa_calor.png", dpi=DPI, bbox_inches="tight")
    plt.show()

def plot_global_correlation_matrix(df, metrics, method, correlation_threshold=0.75):
    correlation_matrix = df[metrics].corr(method=method)
    # mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    figure, axis = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        correlation_matrix,
        # mask=mask,
        cmap="coolwarm",
        fmt='.2f',
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=.5,
        cbar_kws={'label': 'Coeficiente de correlación'},
        ax=axis
    )
    axis.set_title("Matriz de correlación global", fontsize=FONTSIZE)
    axis.tick_params(axis='both', labelsize=FONTSIZE)
    if axis.collections:
        colorbar = axis.collections[0].colorbar
        if colorbar is not None:
            colorbar.ax.tick_params(labelsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "matriz_correlacion_global.png", dpi=DPI)
    plt.show()

    pares = correlation_matrix.unstack().reset_index()
    pares.columns = ['Variable_1', 'Variable_2', 'Correlacion']
    pares = pares.dropna(subset=['Correlacion'])

    pares_altos = pares[
        (pares['Correlacion'].abs() > correlation_threshold) & 
        (pares['Variable_1'] < pares['Variable_2'])
    ].reset_index(drop=True)

    return correlation_matrix, pares_altos


def plot_optimized_distributions(df, topological_metrics, scaling_method):
    metrics_df = df[topological_metrics].copy()
    
    if scaling_method == 'min-max':
        scaler = MinMaxScaler()
    elif scaling_method == 'z-score':
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
        
    scaled_data = scaler.fit_transform(metrics_df)
    scaled_df = pd.DataFrame(scaled_data, columns=metrics_df.columns)
    
    long_df = scaled_df.melt(var_name='Metrica_Topologica', value_name='Valor_Escalado')
    
    figure, axis = plt.subplots(figsize=(14, 6))
    
    sns.violinplot(
        data=long_df,
        x='Metrica_Topologica',
        y='Valor_Escalado',
        hue='Metrica_Topologica',
        legend=False,
        inner="quartile",
        palette="viridis",
        linewidth=1,
        ax=axis
    )
    
    unique_metrics = long_df['Metrica_Topologica'].unique()
    colors = sns.color_palette("viridis", len(unique_metrics))
    
    axis.set_xticks(range(len(unique_metrics)))
    unique_metrics = [metric+"    " for metric in unique_metrics]
    axis.set_xticklabels(unique_metrics, rotation=45, ha='right', color='black')
    
    plt.draw()
    
    for i, (tick, color) in enumerate(zip(axis.get_xticklabels(), colors)):
        axis.annotate(
            "●",
            xy=(i, 0),
            xycoords=('data', 'axes fraction'),
            xytext=tick.get_position(),
            textcoords='offset points',
            color=color,
            rotation=45,
            ha='right',
            va='top',
            fontsize=AXIS_TICK_FONTSIZE
        )
    
    axis.set_title(f"Distribución global de métricas (Escalado: {scaling_method})")
    axis.set_xlabel("Métricas topológicas")
    axis.set_ylabel("Valor transformado")
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / f"distribuciones_globales_{scaling_method}.png", dpi=DPI)
    plt.show()
