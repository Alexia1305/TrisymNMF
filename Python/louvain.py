import networkx as nx
import community as community_louvain  # The Louvain method for community detection
import matplotlib.pyplot as plt
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community.quality import modularity
import pandas as pd

def read_net_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extraire les sommets
    vertices = {}
    edges = []

    reading_vertices = True
    for line in lines:
        if line.startswith('%') or not line.strip():
            continue
        if line.startswith('*Vertices'):
            reading_vertices = True
            continue
        if line.startswith('*Arcs'):

            continue
        elif line.startswith('*Edges'):  # Si on trouve le début des arêtes
            reading_vertices = False
            continue

        if reading_vertices:
            parts = line.split()
            if len(parts) >= 4:
                vertex_id = int(parts[0])
                vertex_label = parts[1].strip('"')
                # On peut ajouter d'autres propriétés si besoin
                vertices[vertex_id] = vertex_label
        else:
            # Ici on traiterait les arêtes (non inclus dans votre extrait)
            # Par exemple, si les arêtes sont dans ce format : "source target"
            source_target = line.split()
            if len(source_target) == 3:
                edges.append((int(source_target[1]), int(source_target[2])))

    # Créer le graphe
    G = nx.Graph()

    # Ajouter les sommets
    for vertex_id, vertex_label in vertices.items():
        G.add_node(vertex_id, label=vertex_label)

    # Ajouter les arêtes
    G.add_edges_from(edges)

    return G
def read_network(file_path):
    """
    Reads a network in Pajek (.net) format and returns a NetworkX graph.
    """
    G = nx.read_pajek(file_path)  # Reads a Pajek format network
    G = nx.Graph(G)  # Convert it to an undirected simple graph
    return G


def find_best_partition_with_louvain(G):
    """
    Finds the best partition using the Louvain method and limits the partition to 2 communities.
    """
    # Apply Louvain method to detect communities
    partition = community_louvain.best_partition(G)

    # Convert partition dict to a list of communities
    communities = {}
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)

    # If more than 2 communities, merge them into 2
    if len(communities) > 2:
        # Sort communities by size
        sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
        # Merge small communities into 2 main ones
        merged_communities = {0: [], 1: []}
        for i, (comm_id, nodes) in enumerate(sorted_communities):
            merged_communities[i % 2].extend(nodes)
        partition = {node: 0 if node in merged_communities[0] else 1 for node in G.nodes()}

    return partition


def plot_partition(G, partition):
    """
    Plots the network graph with nodes colored by their community.
    """
    # Color the nodes by partition
    pos = nx.spring_layout(G)
    cmap = plt.get_cmap('viridis')

    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=[partition[node] for node in G.nodes()],
                           cmap=cmap, node_size=50)
    plt.show()
def plot_communities_with_labels(G, communities):
    """
    Function to plot the network with nodes colored by their community, and labels for each node.
    """
    pos = nx.spring_layout(G)  # Layout for visualization
    colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a']  # Color palette for communities

    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=list(community), node_color=colors[i % len(colors)], node_size=300)

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # Add labels to the nodes
    labels = {node: str(node) for node in G.nodes()}  # Create a dictionary of node labels
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    plt.show()
def plot_communities(G, communities):
    """
    Function to plot the network with nodes colored by their community.
    """
    pos = nx.spring_layout(G)  # Layout for visualization
    colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a']  # Color palette for communities
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=list(community), node_color=colors[i % len(colors)], node_size=100)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

def main():
    # Replace 'network.net' with the path to your .net file
    file_path = '../dataset/Scotland.net'

    # Step 1: Read the network
    G =read_net_file(file_path)
    print("Louvain \t ")
    # Step 2: Find the best partition limited to 2 communities
    partition = find_best_partition_with_louvain(G)

    print(community_louvain.modularity(partition, G))
    # Step 3: Plot the result
    plot_partition(G, partition)

    print("Newman \t")

    communities_generator = girvan_newman(G)
    top_level_communities = next(communities_generator)  # Première partition (2 communautés)

    # Convertir en liste pour être compatible avec le calcul de la modularité
    partition = [list(community) for community in top_level_communities]
    # Étape 3 : Calculer la modularité de la partition trouvée
    print(modularity(G, partition))
    # Step 3: Print the communities
    print("Communities detected:", partition)

    # Step 4: Plot the result
    plot_communities_with_labels(G, partition)




if __name__ == "__main__":
    main()
