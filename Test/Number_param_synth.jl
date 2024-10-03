using MAT
using Printf
using Random
using LightGraphs # Charger le package LightGraphs pour manipuler les graphes
using Plots # Charger le package Plots pour les tracés
using LinearAlgebra
using GraphPlot
using Images

using Statistics

include("../algo/algo_symTriONMF.jl")
include("../algo/ONMF.jl")
include("../algo/symNMF.jl")
include("../algo/TrisymNMF_CD.jl")
include("affichage.jl")
Random.seed!(123)

function calculate_eigenvalue_distribution(n::Int, nbr_tests::Int)
    eigenvalue_distribution = Float64[]

    for i in 1:nbr_tests
        # creation matrice random
        X=rand(n,n)
        for i in 1:n
            for j in 1:n
                if j<i 
                    X[j,i]=X[i,j]
                end 
            end
        end 
        # Calculer les valeurs propres
        eigvals_X=eigen(X).values
        eigenvalues = real(eigvals_X[abs.(imag.(eigvals_X)) .< 1e-20])

        # Ajouter les valeurs propres à la distribution
        append!(eigenvalue_distribution, eigenvalues)
    end

    return eigenvalue_distribution
end

function calcul_v()
    



    # Paramètres
    matrix_size = 200
    nbr_tests = 2000

    # Calculer la distribution des valeurs propres
    distribution = calculate_eigenvalue_distribution(matrix_size, nbr_tests)

    histogram(distribution, bins=30, xlabel="Eigenvalue", ylabel="Frequency", title="Eigenvalue Distribution")

end 

function test()
    r=50
    n=200
    nbr_tests=100

    
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
        print(i)

        # creation matrice random
        X=rand(n,n)
        for k in 1:n
            for l in 1:n
                if l<k
                    X[k,l]=X[l,k]
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

        temps_execution[3,i] = @elapsed begin
            A, erreur = SymNMF(X, r; max_iter=maxiter, max_time=timelimit, tol=epsi, A_init=init)
        end
        erreurs[3,i] = erreur
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
function test_symvstri_sparse(r)

   
    n=200
    nbr_tests=50

    
    ###########OPTIONS##################################################
    init="sspa"
    maxiter=10000
    timelimit=400
    epsi=10e-5
    density=0.1
    lambda=0
    nbr_algo=3

    # Initialisation des tableaux pour stocker les temps et les erreurs
   
    println(r,":")
    Wb=zeros(n,r)
    temps_execution = zeros(nbr_algo,nbr_tests)
    erreurs = zeros(nbr_algo,nbr_tests)
    # Boucle pour effectuer les tests
    for i in 1:nbr_tests
        print(i)

        # creation matrice random
        
        X = abs.(sprandn(n,n,density));
        X = X+X';
      
        # eigvals_X=eigen(X).values
        # println(eigvals_X[abs.(imag.(eigvals_X)) .< 1e-10])

        temps_execution[1,i] = @elapsed begin
            A, erreur = SymNMF(X, r; max_iter=maxiter, max_time=timelimit, tol=epsi, A_init=init)
        end
      
        erreurs[1,i] = erreur
        if r<120
            temps_execution[2,i] = @elapsed begin
                W,S,erreur=TrisymNMF_CD(X, r,lambda, maxiter,epsi,"sspa",timelimit)
            end
            erreurs[2,i] = erreur
        end 
        
        temps_execution[3,i] = @elapsed begin
            Wb, S, erreur = symTriONMF_coordinate_descent(X, r, maxiter, epsi,init, timelimit)
        end
        erreurs[3,i] = erreur
        
    end
    println("")
    # Calcul de la moyenne et de l'écart type des temps et des erreurs
    moyenne_temps = mean(temps_execution, dims=2)
   
    ecart_type_temps = std(temps_execution, dims=2)
    moyenne_erreurs = mean(erreurs, dims=2)
    ecart_type_erreurs = std(erreurs, dims=2)
    # Création du graphique
    methods = ["SymNMF","TrisymNMF","OtrisymNMF"]
    # Affichage des résultats
    for j in 1:nbr_algo
        println("Temps d'exécution pour la méthode ", methods[j], " : ", @sprintf("%.3g", moyenne_temps[j, 1])," +_ ", @sprintf("%.3g", ecart_type_temps[j, 1]), " secondes")
    
        println("l'erreur % pour la méthode ", methods[j], " : ", @sprintf("%.3g", moyenne_erreurs[j, 1]*100)," +_  ",@sprintf("%.3g", ecart_type_erreurs[j, 1]*100)," %")
    end   
    # Création du graphique

    # Enregistrement des résultats dans un fichier texte
    nom_fichier_resultats = "resultats_synth_sparse_$r.txt"
    # Enregistrement des résultats dans un fichier texte
    open(nom_fichier_resultats, "w") do io
       
        write(io, "r = $r\n\n")
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
function test_symvstri_defpos(r)

   
    n=200
    nbr_tests=50

    
    ###########OPTIONS##################################################
    init="sspa"
    maxiter=10000
    timelimit=60
    epsi=10e-9
    density=0.1
    epsilon=0.1
    lambda=0
    nbr_algo=3

    # Initialisation des tableaux pour stocker les temps et les erreurs
   
    println(r,":")
    Wb=zeros(n,r)
    temps_execution = zeros(nbr_algo,nbr_tests)
    erreurs = zeros(nbr_algo,nbr_tests)
    # Boucle pour effectuer les tests
    for i in 1:nbr_tests
        print(i)

        # creation matrice random
        W = sprandn(n,r,density)
        Xclean=W*W'
        N = 2 * rand(n, n) .- 1
        X = max.(0,Xclean+epsilon*N*norm(Xclean)/norm(N))
       
      
        # eigvals_X=eigen(X).values
        # println(eigvals_X[abs.(imag.(eigvals_X)) .< 1e-10])
        if r<130
            temps_execution[1,i] = @elapsed begin
                A, erreur = SymNMF(X, r; max_iter=maxiter, max_time=timelimit, tol=epsi, A_init=init)
            end
            erreurs[1,i] = erreur
        end 
        
        if r<120
            temps_execution[2,i] = @elapsed begin
                W,S,erreur=TrisymNMF_CD(X, r,lambda, maxiter,epsi,"sspa",timelimit)
            end
            erreurs[2,i] = erreur
        end 
        
        temps_execution[3,i] = @elapsed begin
            Wb, S, erreur = symTriONMF_coordinate_descent(X, r, maxiter, epsi,init, timelimit)
        end
        erreurs[3,i] = erreur
        
    end
    println("")
    # Calcul de la moyenne et de l'écart type des temps et des erreurs
    moyenne_temps = mean(temps_execution, dims=2)
   
    ecart_type_temps = std(temps_execution, dims=2)
    moyenne_erreurs = mean(erreurs, dims=2)
    ecart_type_erreurs = std(erreurs, dims=2)
    # Création du graphique
    methods = ["SymNMF","TrisymNMF","OtrisymNMF"]
    # Affichage des résultats
    for j in 1:nbr_algo
        println("Temps d'exécution pour la méthode ", methods[j], " : ", @sprintf("%.3g", moyenne_temps[j, 1])," +_ ", @sprintf("%.3g", ecart_type_temps[j, 1]), " secondes")
    
        println("l'erreur % pour la méthode ", methods[j], " : ", @sprintf("%.3g", moyenne_erreurs[j, 1]*100)," +_  ",@sprintf("%.3g", ecart_type_erreurs[j, 1]*100)," %")
    end   
    # Création du graphique

    # Enregistrement des résultats dans un fichier texte
    nom_fichier_resultats = "resultats_synth_positif_$r.txt"
    # Enregistrement des résultats dans un fichier texte
    open(nom_fichier_resultats, "w") do io
       
        write(io, "r = $r\n\n")
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
liste_r=[100]
for r in liste_r
    test_symvstri_sparse(r)
end 
# calcul_v()


