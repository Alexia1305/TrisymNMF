using MAT
using Statistics
using Plots
using Printf
using Random
using LightGraphs 
using Plots 
using LinearAlgebra
using GraphPlot
include("../algo/TrisymNMF_CD.jl")

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


# # Tests 
# temps_execution = zeros(nbr_algo)
# erreurs = zeros(nbr_algo)

# temps_execution[1] = @elapsed begin
#     W, S, erreur = symTriONMF_coordinate_descent(X, r, maxiter, epsi,init, timelimit)
# end
# erreurs[1] = erreur

# temps_execution[2] = @elapsed begin
#     W_O, H, erreur = alternatingONMF(X, r, maxiter, epsi, init)
# end
# erreurs[2] = erreur

# temps_execution[3] = @elapsed begin
#     A, erreur = SymNMF(X, r; max_iter=maxiter, max_time=timelimit, tol=epsi, A_init=init)
# end
# erreurs[3] = erreur


# # Calcul de la moyenne et de l'écart type des temps et des erreurs

# methods = ["symTriONMF", "ONMF", "SymNMF"]

# # Affichage des résultats
# for j in 1:nbr_algo

#     println("Temps d'exécution pour la méthode ", methods[j], " : ", @sprintf("%.5g", temps_execution[j])," secondes")
    
#     println("l'erreur % pour la méthode ", methods[j], " : ", @sprintf("%.5g", erreurs[j]*100),"  %")
# end   






# club1 = findall(W[:, 1] .> 0)
                
# club2 = findall(W[:, 2] .> 0)
# println("classement par OtrisymNMF")
# println("club1: ",club1)
# println("club2: ",club2)
# # Créer un dictionnaire contenant la matrice W
# variables = Dict("W" => W, "S" => S)

# # Enregistrer le fichier .mat
# matwrite("Karate_OTS.mat", variables)


# club1=[]
# club2=[]

# # Parcourir chaque ligne de la matrice
# for i in 1:size(A, 1)
#     # Trouver l'indice de l'élément le plus grand dans la ligne actuelle
#     a = argmax(A[i, :])
#     if a==1
#         push!(club1,i)
#     else 
#         push!(club2,i)
#     end
# end
# println("classement par ONMF")
# println("club1: ",club1)
# println("club2: ",club2)

# variables=Dict("W"=>A)
# matwrite("Karate_S.mat",variables)

# club1 = findall(H[1, :] .> 0)
                
# club2 = findall(H[2, :] .> 0)
# println("classement par ONMF")
# println("club1: ",club1)
# println("club2: ",club2)

# variables=Dict("W"=>W_O,"H"=>H)
# matwrite("Karate_O.mat",variables)

lambdas=collect(0:0.1:1)
for lambda in lambdas
    temps_execution  = @elapsed begin
        W, S, erreur  =TrisymNMF_CD(X, r,lambda, maxiter,epsi,init,timelimit)
    end
    println(temps_execution," sec")
    prinln(erreur, " %")
    club1 = findall(W[:, 1] .> 0)
                    
    club2 = findall(W[:, 2] .> 0)
    println("classement par trisymNMF")
    println("club1: ",club1)
    println("club2: ",club2)
end 
