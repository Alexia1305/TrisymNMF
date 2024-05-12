using MAT
using Printf
using Random
using LightGraphs # Charger le package LightGraphs pour manipuler les graphes
using Plots # Charger le package Plots pour les tracés
using LinearAlgebra
using GraphPlot
using Images

include("../algo/algo_symTriONMF.jl")
include("../algo/ONMF.jl")
include("../algo/symNMF.jl")
include("affichage.jl")
Random.seed!(123)



function test()
    r=90
    n=200
    nbr_tests=50

    
    ###########OPTIONS##################################################
    init="sspa"
    maxiter=10000
    timelimit=60
    epsi=10e-7
   
    nbr_algo=4

    # Initialisation des tableaux pour stocker les temps et les erreurs
   
    println(n)
    Wb=zeros(n,r)
    temps_execution = zeros(nbr_algo,nbr_tests)
    erreurs = zeros(nbr_algo,nbr_tests)
    # Boucle pour effectuer les tests
    for i in 1:nbr_tests

        # creation matrice random
        X=rand(n,n)
        for i in 1:r
            for j in 1:r 
                if j<i 
                    X[j,i]=X[i,j]
                end 
            end
        end 
        # eigvals_X=eigen(X).values
        # println(eigvals_X[abs.(imag.(eigvals_X)) .< 1e-10])

        temps_execution[1,i] = @elapsed begin
            Wb, S, erreur = symTriONMF_coordinate_descent(X, r, maxiter, epsi,init, timelimit)
        end
        erreurs[1,i] = erreur

        temps_execution[2,i] = @elapsed begin
            W, H, erreur = alternatingONMF(X, r, maxiter, epsi,init)
        end
        erreurs[2,i] = erreur

        # temps_execution[3,i] = @elapsed begin
        #     A, erreur = SymNMF(X, r; max_iter=maxiter, max_time=timelimit, tol=epsi, A_init=init)
        # end
        # erreurs[3,i] = erreur
        temps_execution[4,i] = @elapsed begin
            W,S, erreur = symTriONMF_update_rules(X, r, maxiter, epsi, init,timelimit)
        end
        erreurs[4,i] = erreur
    end

    # Calcul de la moyenne et de l'écart type des temps et des erreurs
    moyenne_temps = mean(temps_execution, dims=2)
   
    ecart_type_temps = std(temps_execution, dims=2)
    moyenne_erreurs = mean(erreurs, dims=2)
    ecart_type_erreurs = std(erreurs, dims=2)
    # Création du graphique
    methods = ["symTriONMF", "ONMF", "SymNMF","MU"]
    # Affichage des résultats
    for j in 1:nbr_algo
        println("Temps d'exécution pour la méthode ", methods[j], " : ", @sprintf("%.3g", moyenne_temps[j, 1])," +_ ", @sprintf("%.3g", ecart_type_temps[j, 1]), " secondes")
    
        println("l'erreur % pour la méthode ", methods[j], " : ", @sprintf("%.3g", moyenne_erreurs[j, 1]*100)," +_  ",@sprintf("%.3g", ecart_type_erreurs[j, 1]*100)," %")
    end   
    # Création du graphique

    # Enregistrement des résultats dans un fichier texte
    nom_fichier_resultats = "resultats_synth.txt"
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

   
    
end 
# # Créer un dictionnaire contenant la matrice W
test()

