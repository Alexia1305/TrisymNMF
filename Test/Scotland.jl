using Graphs
using DelimitedFiles
include("../algo/OtrisymNMF.jl")
function read_pajek(filename::String)
    # Lire tout le fichier
    lines = readlines(filename)
    
    # Trouver la section des sommets
    vertex_index = findfirst(x -> startswith(x, "*Vertices"), lines)
    num_vertices = parse(Int, split(lines[vertex_index])[2])
    
    # Trouver la section des arêtes
    edges_index = findfirst(x -> startswith(x, "*Edges"), lines)
    
    # Créer un graphe avec le nombre de sommets trouvé
    g = SimpleGraph(num_vertices)
    
    # Ajouter les arêtes à partir des lignes qui suivent "*Edges"
    for line in lines[(edges_index+1):end]
        edge = parse.(Int, split(line))
        add_edge!(g, edge[1], edge[2])
    end
    
    return g
end

# Lire le fichier .net et obtenir le graphe
g = read_pajek("dataset/Scotland.net")

# Afficher les nœuds et arêtes du graphe
println("Nodes: ", vertices(g))
println("Edges: ", edges(g))

# Récupérer la matrice d'adjacence
X = float(Matrix(adjacency_matrix(g)))

r=2
init="sspa"
maxiter=10000
timelimit=60
epsi=10e-7

temps_execution = @elapsed begin
    W, S, erreur = OtrisymNMF_CD(X, r, maxiter, epsi,init, timelimit)
end
club1 = findall(W[:, 1] .> 0)
club2= findall(W[:, 2] .> 0)
println(size(club1))
println(size(club2))
println("number of edges inter communities",count(x -> x != 0, X[club1,club1])+count(x -> x != 0, X[club2,club2]))
