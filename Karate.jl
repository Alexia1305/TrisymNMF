using MAT
file_path = "dataset/karate.mat"
include("algo_symTriONMF.jl")



using LightGraphs # Charger le package LightGraphs pour manipuler les graphes
using Plots # Charger le package Plots pour les tracés
using LinearAlgebra

using GraphPlot
# Charger le fichier karate.mat
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
println(X)
# Rang interne de la factorisation
r = 2

# Options de symNMF (voir également loadoptions.m)
maxiter=10000
timelimit=5
epsi=10e-7
# Exécution de symNMF
temps_execution_1 = @elapsed begin
    W, S, erreur = symTriONMF_coordinate_descent(X, r, maxiter,epsi,true,timelimit)
end

println(W)
println(S)
println(erreur)


club1 = findall(W[:, 1] .> 0)
                
club2 = findall(W[:, 2] .> 0)

println(club1)
println(club2)
println(S)
println(W)
# Créer un dictionnaire contenant la matrice W
variables = Dict("W" => W)

# Enregistrer le fichier .mat
matwrite("W.mat", variables)
# Créer un dictionnaire contenant la matrice W
variableS= Dict("S" => S)


# Enregistrer le fichier .mat
matwrite("S.mat", variableS)