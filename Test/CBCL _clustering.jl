using MAT
using Printf
using Random
using LightGraphs # Charger le package LightGraphs pour manipuler les graphes
using Plots # Charger le package Plots pour les tracés
using LinearAlgebra
using GraphPlot
using Images
using ColorSchemes

include("../algo/algo_symTriONMF.jl")
include("../algo/ONMF.jl")
include("../algo/symNMF.jl")
include("affichage.jl")
Random.seed!(123)




# Charger le fichier karate.mat
file_path = "dataset/CBCL.mat"
mat = matread(file_path)
A = mat["X"]
A=A[:,1:50]
indices_melanges = shuffle(1:size(A, 2))
Person=A[:,indices_melanges]

matrice_img=affichage(Person,10,19,19,1)
file_name="CBCL_50.png"
save(file_name,matrice_img)

X=Person'*Person
###########OPTIONS##################################################
r = 11
init="sspa"
maxiter=10000
timelimit=5
epsi=10e-7

temps_execution = @elapsed begin
    W, S, erreur = symTriONMF_coordinate_descent(X, r, maxiter, epsi,init, timelimit)
end
classement=Vector{Vector{Int}}(undef, r)  # Initialisation d'un vecteur de r éléments
global vide=0
for i=1:r
    
    classement[i] = findall(W[:, i] .> 0)
    if size(classement[i])[1]>0
        matrice_img=affichage(Person[:,classement[i]],size(classement[i])[1],19,19,1)
        file_name="CBCL_$i.png"
        save(file_name,matrice_img)
    else
        global vide=vide+1
        println("classe vide ")   
    end          
end 
println(vide)

matrice=S./norm(S,2)
for i in 1:r
    matrice[i,i]=0
end
# Obtenir la palette de couleurs en niveaux de gris
grays = cgrad(:grays)

# Inverser l'ordre des couleurs pour obtenir des niveaux de gris inversés
inverted_grays = reverse(grays)

# Créer l'objet carte thermique avec la palette de couleurs inversée
heatmap_obj = heatmap(matrice, color=inverted_grays)

# Afficher la carte thermique
plot(heatmap_obj)

