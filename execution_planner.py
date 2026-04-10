import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple

class ExecutionPlanner:
    """
    Groups updates into parallel execution layers and provides DAG analytics.
    """

    def plan_layers(self, n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)

        # topological_generations returns sets of nodes that can be run in parallel
        return [list(layer) for layer in nx.topological_generations(G)]

    def visualize_dag(self, n: int, edges: List[Tuple[int, int]], save_path: str):
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)

        if G.number_of_nodes() > 100:
            print("Skipping visualization: Graph too large.")
            return

        plt.figure(figsize=(12, 8))
        # Assign 'layer' attribute for multipartite layout
        for i, layer in enumerate(nx.topological_generations(G)):
            for node in layer:
                G.nodes[node]['layer'] = i
        
        pos = nx.multipartite_layout(G, subset_key="layer")
        nx.draw(G, pos, with_labels=True, node_color='skyblue', 
                node_size=400, arrowsize=15, font_size=8)
        
        plt.title(f"Memory Dependency DAG\n(Parallel Layers: {len(list(nx.topological_generations(G)))})")
        plt.savefig(save_path)
        plt.close()