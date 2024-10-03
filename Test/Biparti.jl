using Graphs  # Remplace LightGraphs par Graphs
using GraphPlot
using Colors
using Plots
include("../algo/OtrisymNMF.jl")

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

# AFFICHAGE 


# Création du graphe à partir de la matrice d'adjacence
g = SimpleGraph(X)
# Calculer les positions
pos = spring_layout(g)
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
    x = [pos[1][src(e)], pos[1][dst(e)]]  # Coordonnées X des arêtes
    y = [pos[2][src(e)], pos[2][dst(e)]]  # Coordonnées Y des arêtes
    plot!(x, y, color=:gray,label="")
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

    scatter!([pos[1][i]], [pos[2][i]], marker=shape, label="", color=node_colors[i], markersize=8)
end

# Afficher le graphe
xlabel!("X")
ylabel!("Y")
title!("Graph with Different Node Shapes")

