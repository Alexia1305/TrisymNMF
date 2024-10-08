using Graphs
using DelimitedFiles
include("../algo/OtrisymNMF.jl")
include("tools_CD.jl")
include("affiche.jl")
 # Remplace LightGraphs par Graphs

using Colors
using LightGraphs 
using Plots
function test_Scotland()
    # Lire le fichier .net et obtenir le graphe
    g = read_pajek("dataset/Scotland.net")

   

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
    n,r=size(W)
    partition=zeros(n)
    for i in 1:n
        # Trouver l'indice de la colonne où l'élément est non nul
        partition[i] = findfirst(x -> x != 0, W[i, :])
    end
    println("modularity",Modularity(X,partition))
end 
function Dolphins()

    g,labels = read_pajek("dataset/dolphins.net")
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
    println("number of edges out communities :",count(x -> x != 0, X[club1,club2]))
    n,r=size(W)
    partition=zeros(n)
    for i in 1:n
        # Trouver l'indice de la colonne où l'élément est non nul
        partition[i] = findfirst(x -> x != 0, W[i, :])
    end
    println("modularity: ",Modularity(X,partition))
    True_partition=ones(n)
    for i in [61,33,57,23,6,10,7,32,14,18,26,49,58,43,55,28,27,2,20,8]
        True_partition[i]=2
    end 
    println("Inorm : ",normalised_mutual_info(True_partition,partition))
    # Définir les couleurs en fonction des classes
    # Par exemple, classe 1 => bleu, classe 2 => rouge
    nodecolor = [colorant"lightseagreen", colorant"orange"]
    nodefillc = nodecolor[Int.(partition)]
    # Générer les positions des noeuds (disposition par ressort)
    pos = spring_layout(g)
    # Appliquer une transformation pour espacer davantage les nœuds
    
    
    # Créer une figure
    p=plot()

    # Tracer les arêtes
    for e in edges(g)
        x = [pos[1][src(e)], pos[1][dst(e)]]  # Coordonnées X des arêtes
        y = [pos[2][src(e)], pos[2][dst(e)]]  # Coordonnées Y des arêtes
        plot!(x, y, color=:gray,linewidth=0.5,label="", axis=false,grid=false)
    end

    # Tracer les nœuds avec différentes formes
    for i in 1:nv(g)
        scatter!([pos[1][i]], [pos[2][i]], label="", color=nodefillc[i], markersize=4)
    end

    # # Ajouter une légende
    # scatter!(1, 1, label="Women", color=:red, ms=10, leg=:topright)
    # scatter!(2, 1, label="Events", color=:blue, ms=10)
    # Extraire les coordonnées des positions
    x_positions = [pos[1][i] for i in 1:nv(g)]
    y_positions = [pos[2][i] for i in 1:nv(g)]

    # Créer les annotations (étiquettes des noeuds, positionnées légèrement en dessous)
    
    annotations = [(x_positions[i], y_positions[i] - 0.05, string(labels[i])) for i in 1:nv(g)]  # Décaler les étiquettes
    # Afficher le graphe
    scatter!(x_positions, y_positions, annotations=annotations, ms=0,label="", annotationfontsize=6)  # Ajouter les labels
    # Afficher le graphe
    display(p)
    affiche(S,"S_dolphins")




end

Dolphins()
