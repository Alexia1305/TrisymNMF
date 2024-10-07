using Graphs  # Remplace LightGraphs par Graphs
using GraphPlot
using Colors
using Plots
include("../algo/OtrisymNMF.jl")
include("Bi_modularity.jl")

A = [
    1 1 1 1 1 1 0 1 1 0 0 0 0 0;
    1 1 1 0 1 1 1 1 0 0 0 0 0 0;
    0 1 1 1 1 1 1 1 1 0 0 0 0 0;
    1 0 1 1 1 1 1 1 0 0 0 0 0 0;
    0 0 1 1 1 0 1 0 0 0 0 0 0 0;
    0 0 1 0 1 1 0 1 0 0 0 0 0 0;
    0 0 0 0 1 1 1 1 0 0 0 0 0 0;
    0 0 0 0 0 1 0 1 1 0 0 0 0 0;
    0 0 0 0 1 0 1 1 1 0 0 0 0 0;
    0 0 0 0 0 0 1 1 1 0 0 1 0 0;
    0 0 0 0 0 0 0 1 1 1 0 1 0 0;
    0 0 0 0 0 0 0 1 1 1 0 1 1 1;
    0 0 0 0 0 0 1 1 1 1 0 1 1 1;
    0 0 0 0 0 1 1 0 1 1 1 1 1 1;
    0 0 0 0 0 0 1 1 0 1 1 1 0 0;
    0 0 0 0 0 0 0 1 1 0 0 0 0 0;
    0 0 0 0 0 0 0 0 1 0 1 0 0 0;
    0 0 0 0 0 0 0 0 1 0 1 0 0 0
]

p,q=size(A)
# Afficher la matrice
println(A)
n=p+q
X=zeros(n,n)
for i in 1:p
    for j in 1:q
        if A[i,j]==1
            X[i,p+j]=1
            X[p+j,i]=1
        end
    end
end
########## OPTIONS  ##############
r=8
init="sspa"
maxiter=10000
timelimit=30
epsi=10e-7
erreur=0
# Tests 

temps_execution = @elapsed begin
    W, S, erreur = OtrisymNMF_CD(X, r, maxiter, epsi,init, timelimit)
end

println('w', W)
println('S', S)
println("error", erreur)
for k in 1:r
    club = findall(W[:, k] .> 0)
    print("Club ",k, " : ",club)
    
end 
node_classes=zeros(n)
for i in 1:n
    node_classes[i]=findall(W[i, :] .> 0)[1]
end 
delta=zeros(n,n)
for k in 1:r

    groups_of_2 = collect(combinations(findall(W[:,k] .> 0), 2))
    for (i,j) in groups_of_2
         delta[i,j] =1
    end 
end 
println(BI_Modularity(X,delta))

# AFFICHAGE 


# Création du graphe à partir de la matrice d'adjacence
g = SimpleGraph(X)
# # Calculer les positions
# Trouver les nœuds appartenant à chaque classe
unique_classes = unique(node_classes)  # Trouver les classes uniques
class_groups = [findall(x -> x == c, node_classes) for c in unique_classes]  # Grouper les nœuds par classe

# Assigner des positions aléatoires par cluster
cluster_radius = 0.2  # Rayon pour positionner les nœuds dans le même cluster
cluster_centers = [(cos(2*pi*i/length(unique_classes)), sin(2*pi*i/length(unique_classes))) for i in 1:length(unique_classes)]  # Centres des clusters autour d'un cercle

# Fonction pour générer des positions proches autour d'un centre de cluster
function generate_cluster_positions(center, num_nodes, radius)
    θ = LinRange(0, 2*pi, num_nodes+1)[1:end-1]  # Angles également espacés
    [(center[1] + radius * cos(θ[i]), center[2] + radius * sin(θ[i])) for i in 1:num_nodes]
end
# Générer les positions des nœuds par classe
pos = Dict()
for (i, class_group) in enumerate(class_groups)
    cluster_center = cluster_centers[i]
    cluster_positions = generate_cluster_positions(cluster_center, length(class_group), cluster_radius)
    for (j, node) in enumerate(class_group)
        pos[node] = cluster_positions[j]
    end
end
# pos = spring_layout(g)
# Créer une figure
plot()

# Définir les couleurs selon les groupes 1 à 18 et 19 à 32
node_colors = [i <= 18 ? "red" : "blue" for i in 1:nv(g)]

# Définir les formes en fonction des classes 1 à 8
# Par exemple, on pourrait faire 4 formes différentes :
# :circle, :rect, :diamond, :cross (ce sont des symboles Plot)

node_shapes = [node_classes[i] == 1 ? :circle :
               node_classes[i] == 2 ? :rect :
               node_classes[i] == 3 ? :diamond :
               node_classes[i] == 4 ? :cross :
               node_classes[i] == 5 ? :circle :
               node_classes[i] == 6 ? :rect :
               node_classes[i] == 7 ? :diamond :
               cross for i in 1:nv(g) ]

#= # Tracer le graphe avec GraphPlot
gplot(g,
    nodefillc=node_colors,
    nodelabel=1:nv(g),       # Afficher les numéros des nœuds
    nodesize=0.3
) =#

# Tracer les arêtes
for e in edges(g)
    x = [pos[src(e)][1], pos[dst(e)][1]]  # Coordonnées X des arêtes
    y = [pos[src(e)][2], pos[dst(e)][2]]  # Coordonnées Y des arêtes
    plot!(x, y, color=:gray,label="", axis=false,grid=false)
end

# Tracer les nœuds avec différentes formes
for i in 1:nv(g)
    shape = node_classes[i] == 1 ? :circle :
    node_classes[i] == 2 ? :circle :
    node_classes[i] == 3 ? :diamond :
    node_classes[i] == 4 ? :diamond :
    node_classes[i] == 5 ? :cross :
    node_classes[i] == 6 ? :rect :
    node_classes[i] == 7 ? :cross : :rect # Différentes formes

    scatter!([pos[i][1]], [pos[i][2]], marker=shape, label="", color=node_colors[i], markersize=8)
end

# # Ajouter une légende
# scatter!(1, 1, label="Women", color=:red, ms=10, leg=:topright)
# scatter!(2, 1, label="Events", color=:blue, ms=10)
# Extraire les coordonnées des positions
x_positions = [pos[i][1] for i in 1:nv(g)]
y_positions = [pos[i][2] for i in 1:nv(g)]

# Créer les annotations (étiquettes des noeuds, positionnées légèrement en dessous)
labels = collect(1:nv(g))  # Numérote les noeuds de 1 à n
annotations = [(x_positions[i], y_positions[i] - 0.10, string(labels[i])) for i in 1:nv(g)]  # Décaler les étiquettes
# Afficher le graphe
scatter!(x_positions, y_positions, annotations=annotations, ms=0,label="", annotationfontsize=8)  # Ajouter les labels



