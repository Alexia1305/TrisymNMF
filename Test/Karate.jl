using MAT
using Statistics
using Plots
using Printf
using Random
using LightGraphs 
using Plots 
using LinearAlgebra
using GraphPlot

include("../algo/algo_symTriONMF.jl")
include("../algo/ONMF.jl")
include("../algo/symNMF.jl")

Random.seed!(123)

# Charger le fichier karate.mat
file_path = "dataset/karate.mat"
mat = matread(file_path)
edges = mat["edges"]

# Construction de la matrice X
edg = size(edges, 1)
n = Int(maximum(edges))
X = diagm(ones(n)) # Matrice identité de taille n
for i in 1:Int(edg)
    X[Int(edges[i, 1]), Int(edges[i, 2])] = 1
    X[Int(edges[i, 2]), Int(edges[i, 1])] = 1
end


########## OPTIONS  ##############
r=2
init="sspa"
maxiter=10000
timelimit=5
epsi=10e-7
nbr_tests=20
nbr_algo=3


# Tests 
temps_execution = zeros(nbr_algo,nbr_tests)
erreurs = zeros(nbr_algo,nbr_tests)
for i in 1:nbr_tests
    temps_execution[1,i] = @elapsed begin
        W, S, erreur = symTriONMF_coordinate_descent(X, r, maxiter, epsi,init, timelimit)
    end
    erreurs[1,i] = erreur

    temps_execution[2,i] = @elapsed begin
        W, H, erreur = alternatingONMF(X, r, maxiter, epsi, init)
    end
    erreurs[2,i] = erreur

    temps_execution[3,i] = @elapsed begin
        A, erreur = SymNMF(X, r; max_iter=maxiter, max_time=timelimit, tol=epsi, A_init=init)
    end
    erreurs[3,i] = erreur
end

# Calcul de la moyenne et de l'écart type des temps et des erreurs
moyenne_temps = mean(temps_execution, dims=2)
ecart_type_temps = std(temps_execution, dims=2)
moyenne_erreurs = mean(erreurs, dims=2)
ecart_type_erreurs = std(erreurs, dims=2)
methods = ["symTriONMF", "ONMF", "SymNMF"]

# Affichage des résultats
for j in 1:nbr_algo

    println("Temps d'exécution pour la méthode ", methods[j], " : ", @sprintf("%.3g", moyenne_temps[j, 1])," +_ ", @sprintf("%.3g", ecart_type_temps[j, 1]), " secondes")
    
    println("l'erreur % pour la méthode ", methods[j], " : ", @sprintf("%.3g", moyenne_erreurs[j, 1]*100)," +_  ",@sprintf("%.3g", ecart_type_erreurs[j, 1]*100)," %")
end   


# Enregistrement des résultats dans un fichier texte
nom_fichier_resultats = "resultats_Karate.txt"
open(nom_fichier_resultats, "w") do io
    write(io, "Paramètres :\n")
    write(io, "maxiter = $maxiter\n")
    write(io, "timelimit = $timelimit\n")
    write(io, "epsi = $epsi\n")
    write(io, "init = $init\n")
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
# figures 
scatter_plot_temps=scatter(methods, moyenne_temps[:, 1], yerr=ecart_type_temps[:, 1], label="Temps d'exécution moyen ± écart-type", xlabel="Méthode", ylabel="Temps d'exécution (s)", title="Comparaison des méthodes sur KARATE")
scatter_plot_erreurs=scatter(methods, moyenne_erreurs[:, 1], yerr=ecart_type_erreurs[:, 1], label="Erreur moyenne ± écart-type", xlabel="Méthode", ylabel="Erreur", title="Comparaison des méthodes sur KARATE")
savefig(scatter_plot_temps, "Karate_temps.png")
savefig(scatter_plot_erreurs, "Karate_erreur.png")


# club1 = findall(W[:, 1] .> 0)
                
# club2 = findall(W[:, 2] .> 0)

# # Créer un dictionnaire contenant la matrice W
# variables = Dict("W" => W)

# # Enregistrer le fichier .mat
# matwrite("W.mat", variables)
# # Créer un dictionnaire contenant la matrice W
# variableS= Dict("S" => S)


# # Enregistrer le fichier .mat
# matwrite("S.mat", variableS)