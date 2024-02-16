import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
# création matrice
import h5py

# Ouvrez le fichier .mat en mode lecture
with h5py.File('X.mat', 'r') as file:
    # Accédez à la matrice X (ou tout autre variable que vous souhaitez charger)
    X = file['X'][:]

S=[[6.65470487,0.93634964],[0.93634964,6.65383964]]
import numpy as np




print(X)
# Récupérer la taille de la matrice X
n = X.shape[0]

# Créer une matrice identité de la même taille que X
identity_matrix = np.eye(n)

# Retirer l'identité de X
X_without_identity = X - identity_matrix

# Créer un objet graphique à partir de la matrice d'adjacence
G = nx.from_numpy_matrix(X_without_identity)
D
# Définir les classes
class_1 = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18, 20, 22]
class_2 = [9, 10, 15, 16, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

# Ajouter les nœuds avec les attributs de couleur
for node in G.nodes():
    if node+1 in class_1:
        G.nodes[node]['color'] = 'blue'
    elif node+1 in class_2:
        G.nodes[node]['color'] = 'red'
    else:
        G.nodes[node]['color'] = 'gray'  # Pour les nœuds qui ne sont pas dans les classes définies

# Récupérer les couleurs des nœuds
node_colors = [G.nodes[node]['color'] for node in G.nodes()]


# Définir l'épaisseur des arêtes en fonction des poids
edge_widths = [X[i, j] * 5 for i, j in G.edges()]  # Vous pouvez ajuster le facteur multiplicatif selon vos préférences
# Visualiser le graphe avec les couleurs des clusters
pos = nx.spring_layout(G)  # Vous pouvez utiliser différents algorithmes de disposition
# Définir l'épaisseur des arêtes en fonction des poids
edge_widths = [X[i, j] * 1 for i, j in G.edges()]  # Vous pouvez ajuster le facteur multiplicatif selon vos préférences

# Visualiser le graphe avec l'épaisseur des arêtes sans couleurs
nx.draw_networkx_edges(G, pos, width=edge_widths)
nx.draw_networkx_nodes(G, pos, node_size=700,node_color=node_colors)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
plt.show()

