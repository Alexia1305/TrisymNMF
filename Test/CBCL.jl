using MAT
file_path = "dataset/CBCL.mat"
include("../algo/algo_symTriONMF.jl")
include("../algo/ONMF.jl")
include("../algo/symNMF.jl")

using Printf
using Random
Random.seed!(123)
using LightGraphs # Charger le package LightGraphs pour manipuler les graphes
using Plots # Charger le package Plots pour les tracés
using LinearAlgebra

using GraphPlot
using Images

function affichage(W::Matrix{Float64})
    # Créer une image vide
    largeur, hauteur = 19, 19
    image_combinee = zeros(RGB, hauteur, largeur * size(W, 2))

    # Redimensionner et combiner les images générées par les colonnes de W
    for i in 1:size(W, 2)
        # Remettre la colonne de W en matrice 19x19
        image_colonne = reshape(W[:, i], largeur, hauteur)

        # Normaliser les valeurs entre 0 et 1
        image_colonne .= (image_colonne .- minimum(image_colonne)) / (maximum(image_colonne) - minimum(image_colonne))
        image_colonne=abs.(1 .- image_colonne)
        # Ajouter l'image à l'image combinée
        image_combinee[:, (i - 1) * largeur + 1:i * largeur] .= Gray.(image_colonne)
    end

    # Afficher l'image combinée
    return image_combinee
end
function test()
    # Charger le fichier karate.mat
    mat = matread(file_path)
    A = mat["X"]

    X=A*A'
    # Rang interne de la factorisation
    r = 5
    n=size(X)[1]
    Wb=zeros(n,r)
    # Options de symNMF (voir également loadoptions.m)
    maxiter=10000
    timelimit=5
    epsi=10e-7
    nbr_tests=20
    nbr_algo=3
    # Initialisation des tableaux pour stocker les temps et les erreurs
    temps_execution = zeros(nbr_algo,nbr_tests)
    erreurs = zeros(nbr_algo,nbr_tests)
    # Boucle pour effectuer les tests
    for i in 1:nbr_tests
        temps_execution[1,i] = @elapsed begin
            Wb, S, erreur = symTriONMF_coordinate_descent(X, r, maxiter, epsi, "k_means", timelimit)
        end
        erreurs[1,i] = erreur

        temps_execution[2,i] = @elapsed begin
            W, H, erreur = alternatingONMF(X, r, maxiter, epsi, "k_means")
        end
        erreurs[2,i] = erreur

        temps_execution[3,i] = @elapsed begin
            A, erreur = SymNMF(X, r; max_iter=maxiter, max_time=timelimit, tol=epsi, A_init="k_means")
        end
        erreurs[3,i] = erreur
    end

    # Calcul de la moyenne et de l'écart type des temps et des erreurs
    moyenne_temps = mean(temps_execution, dims=2)
    ecart_type_temps = std(temps_execution, dims=2)
    moyenne_erreurs = mean(erreurs, dims=2)
    ecart_type_erreurs = std(erreurs, dims=2)
    # Création du graphique
    methods = ["symTriONMF", "ONMF", "SymNMF"]
    # Affichage des résultats
    for j in 1:nbr_algo
        println("Temps d'exécution pour la méthode ", methods[j], " : ", @sprintf("%.3g", moyenne_temps[j, 1])," +_ ", @sprintf("%.3g", ecart_type_temps[j, 1]), " secondes")
    
        println("l'erreur % pour la méthode ", methods[j], " : ", @sprintf("%.3g", moyenne_erreurs[j, 1]/100)," +_  ",@sprintf("%.3g", ecart_type_erreurs[j, 1]/100)," %")
    end   
    # Création du graphique

    # Enregistrement des résultats dans un fichier texte
    nom_fichier_resultats = "resultats_CBCL.txt"
    # Enregistrement des résultats dans un fichier texte
    open(nom_fichier_resultats, "w") do io
        write(io, "Paramètres :\n")
        write(io, "maxiter = $maxiter\n")
        write(io, "timelimit = $timelimit\n")
        write(io, "epsi = $epsi\n")
        write(io, "nbr_tests = $nbr_tests\n\n")
        write(io, "Moyennes des temps d'exécution :\n")
        write(io, "$methods\n")
        write(io, join(@sprintf("%.3g", x) for x in vec(moyenne_temps)) * "\n\n")
        write(io, "Écart types des temps d'exécution :\n")
        write(io, "$methods\n")
        write(io, join(@sprintf("%.3g", x) for x in vec(ecart_type_temps)) * "\n\n")
        write(io, "Moyennes des erreurs :\n")
        write(io, "$methods\n")
        write(io, join(@sprintf("%.3g", x) for x in vec(moyenne_erreurs)) * "\n\n")
        write(io, "Écart types des erreurs :\n")
        write(io, "$methods\n")
        write(io, join(@sprintf("%.3g", x) for x in vec(ecart_type_erreurs)) * "\n")
    end

    scatter_plot_temps=scatter(methods, moyenne_temps[:, 1], yerr=ecart_type_temps[:, 1], label="Temps d'exécution moyen ± écart-type", xlabel="Méthode", ylabel="Temps d'exécution (s)", title="Comparaison des méthodes sur KARATE")
    scatter_plot_erreurs=scatter(methods, moyenne_erreurs[:, 1], yerr=ecart_type_erreurs[:, 1], label="Erreur moyenne ± écart-type", xlabel="Méthode", ylabel="Erreur", title="Comparaison des méthodes sur KARATE")
    savefig(scatter_plot_temps, "CBCL_temps.png")
    savefig(scatter_plot_erreurs, "CBCL_erreur.png")
    return Wb
end 
# # Créer un dictionnaire contenant la matrice W
Wb=test()
print(Wb)
variables = Dict("W" => Wb)

# Enregistrer le fichier .mat
matwrite("WCBCL.mat", variables)
# # Créer un dictionnaire contenant la matrice W
# variableS= Dict("S" => S)


# # Enregistrer le fichier .mat
# matwrite("Sc.mat", variableS)
# # Afficher la heatmap
# # Afficher la heatmap


# display(affichage(W))

# # Convertir l'affichage en une image
