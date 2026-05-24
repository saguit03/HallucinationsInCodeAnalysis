import networkx as nx
import pandas as pd
import numpy as np

def build_hallucination_graph_from_run(exec_logs, qwen_results):
    G = nx.DiGraph()
    if isinstance(exec_logs, dict):
        logs = exec_logs.get('logs')
    elif isinstance(exec_logs, list):
        logs = exec_logs
    else:
        logs = []

    for log in (logs or []):
        if not isinstance(log, dict):
            continue
        if log.get('event_type') != 'EDGE_PROCESS':
            continue
        node_id = log.get('node_id')
        details = log.get('details')
        if not node_id or not details:
            continue
        target = details.get('to_node')
        if target:
            G.add_edge(node_id, target)

    qwen_df = pd.DataFrame(qwen_results or [])
    hallucinated_nodes = set()
    if not qwen_df.empty and 'score' in qwen_df.columns and 'node_id' in qwen_df.columns:
        hallucinated_nodes = set(qwen_df[qwen_df['score'] == 'HALLUCINATION DETECTED']['node_id'].unique())

    in_degree = dict(G.in_degree())
    betweenness = nx.betweenness_centrality(G)
    try:
        pagerank = nx.pagerank(G)
    except Exception:
        pagerank = {n: 0.0 for n in G.nodes()}

    for node in G.nodes():
        nx.set_node_attributes(G, {
            node: {
                'hallucinated': node in hallucinated_nodes,
                'in_degree': in_degree.get(node, 0),
                'betweenness': betweenness.get(node, 0.0),
                'pagerank': pagerank.get(node, 0.0)
            }
        })

    metrics_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index').reset_index()
    if not metrics_df.empty:
        metrics_df.rename(columns={'index': 'node_id'}, inplace=True)

    return G, metrics_df

def build_all_hallucination_graphs(warehouse_dataframe):
    graphs = {}
    for _, run_row in warehouse_dataframe.iterrows():
        run_name = run_row.get('run_name') or run_row.get('run_path') or f'run_{_}'
        exec_logs = run_row.get('execution_logs')
        qwen_results = run_row.get('qwen_results')
        G, metrics = build_hallucination_graph_from_run(exec_logs, qwen_results)
        graphs[run_name] = {'graph': G, 'metrics': metrics}
    return graphs

def _safe_stat(values, reducer, default=np.nan):
    cleaned = [value for value in values if pd.notna(value)]
    if not cleaned:
        return default
    try:
        return float(reducer(cleaned))
    except Exception:
        return default

def get_info_directed_graph(grafo):
    metrics = {'status': 'error'}
    
    if grafo is None or grafo.number_of_nodes() == 0:
        return metrics
    
    try:
        metrics = {}
        n_nodes = grafo.number_of_nodes()
        n_edges = grafo.number_of_edges()
        metrics['nodos'] = n_nodes
        metrics['aristas'] = n_edges
        metrics['dirigido'] = nx.is_directed(grafo)
        metrics['densidad'] = float(nx.density(grafo))
        
        in_degrees = [degree for _, degree in grafo.in_degree()]
        out_degrees = [degree for _, degree in grafo.out_degree()]
        total_degrees = [degree for _, degree in grafo.degree()]
        
        metrics['in_degree_medio'] = _safe_stat(in_degrees, np.mean)
        metrics['out_degree_medio'] = _safe_stat(out_degrees, np.mean)
        metrics['grado_total_medio'] = _safe_stat(total_degrees, np.mean)
        metrics['in_degree_max'] = float(max(in_degrees)) if in_degrees else 0.0
        metrics['out_degree_max'] = float(max(out_degrees)) if out_degrees else 0.0
        metrics['in_degree_min'] = float(min(in_degrees)) if in_degrees else 0.0
        metrics['out_degree_min'] = float(min(out_degrees)) if out_degrees else 0.0
        
        if grafo.is_directed():
            metrics['reciprocidad'] = float(nx.reciprocity(grafo))
            metrics['scc_count'] = int(nx.number_strongly_connected_components(grafo))
            metrics['wcc_count'] = int(nx.number_weakly_connected_components(grafo))
            metrics['fuertemente_conexo'] = bool(nx.is_strongly_connected(grafo))
            
            if grafo.number_of_nodes() > 1:
                try:
                    if metrics.get('fuertemente_conexo', False):
                        metrics['diametro'] = int(nx.diameter(grafo))
                        metrics['camino_medio'] = float(nx.average_shortest_path_length(grafo))
                    else:
                        sccs = list(nx.strongly_connected_components(grafo))
                        if sccs:
                            largest_scc_nodes = max(sccs, key=len)
                            subgraph = grafo.subgraph(largest_scc_nodes).copy()
                            if subgraph.number_of_nodes() > 1:
                                metrics['diametro_scc_max'] = int(nx.diameter(subgraph))
                                metrics['camino_medio_scc_max'] = float(nx.average_shortest_path_length(subgraph))
                            else:
                                metrics['diametro_scc_max'] = np.nan
                                metrics['camino_medio_scc_max'] = np.nan
                except Exception:
                    metrics['diametro'] = np.nan
                    metrics['camino_medio'] = np.nan
            
            clustering = nx.average_clustering(grafo.to_undirected())
            metrics['clustering_medio'] = float(clustering)
            assortativity = nx.degree_assortativity_coefficient(grafo.to_undirected())
            metrics['asortatividad'] = float(assortativity)
            scc_sizes = sorted((len(component) for component in nx.strongly_connected_components(grafo)), reverse=True)
            metrics['scc_mayor_tamaño'] = int(scc_sizes[0]) if scc_sizes else 0
            cycles = sum(1 for _ in nx.simple_cycles(grafo))
            metrics['ciclos_simples'] = int(cycles)
        else:
            metrics['componentes_conexas'] = int(nx.number_connected_components(grafo))
        
        node_attributes = pd.DataFrame.from_dict(dict(grafo.nodes(data=True)), orient='index')
        
        hallucinated_ratio = node_attributes['hallucinated'].mean()
        metrics['proporcion_alucinados'] = float(hallucinated_ratio)
        
        hallucinated_metrics = node_attributes[node_attributes['hallucinated'] == True]
        normal_metrics = node_attributes[node_attributes['hallucinated'] == False]
        
        metrics['betweenness_alucinados_media'] = float(hallucinated_metrics['betweenness'].mean())
        metrics['in_degree_alucinados_media'] = float(hallucinated_metrics['in_degree'].mean())
        metrics['pagerank_alucinados_media'] = float(hallucinated_metrics['pagerank'].mean())

        metrics['betweenness_normales_media'] = float(normal_metrics['betweenness'].mean())
        metrics['in_degree_normales_media'] = float(normal_metrics['in_degree'].mean())
        metrics['pagerank_normales_media'] = float(normal_metrics['pagerank'].mean())
        
        metrics['status'] = 'success'
        return metrics
    
    except Exception as e:
        metrics['status'] = 'error'
        metrics['error'] = str(e)
        return metrics
        

        
def get_all_graph_metrics(graphs_dict):
    metrics_list = []
    for run_name, graph_data in graphs_dict.items():
        grafo = graph_data.get('graph')
        if grafo:
            metrics = get_info_directed_graph(grafo)
            metrics['run_name'] = run_name
            metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df