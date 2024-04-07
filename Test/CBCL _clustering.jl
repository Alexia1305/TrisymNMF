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

nbr_images=100
if nbr_images==50
    r=10
    groupe = [
        collect(1:7),
        collect(8:17),
        collect(18:20),
        collect(21:24),
        collect(25:28),
        [29, 30, 33:37...], # Utilisation de ... pour étendre l'intervalle
        [31, 32, 38:40...], # Utilisation de ... pour étendre l'intervalle
        collect(41:44),
        collect(45:48),
        [49, 50]
    ]
else 
    nbr_images=100
    r=19
    groupe=[
        collect(1:7),
        collect(8:17),
        collect(18:20),
        collect(21:24),
        collect(25:28),
        [29, 30, 33:37...], # Utilisation de ... pour étendre l'intervalle
        [31, 32, 38:40...], # Utilisation de ... pour étendre l'intervalle
        collect(41:44),
        collect(45:48),
        collect(45:53),
        collect(54:56),
        collect(57:63),
        collect(64:72),
        collect(73:77),
        collect(78:81),
        collect(82:87),
        collect(88:93),
        collect(94:97),
        collect(98:100)

    ]
end
W_true=zeros(nbr_images,r)
for (indice, element) in enumerate(groupe)
    W_true[element,indice] .= 1
end 

# Charger le fichier karate.mat
file_path = "dataset/CBCL.mat"
mat = matread(file_path)
A = mat["X"]
A=A[:,1:nbr_images]

# # prepocessing
# for i in 1: nbr_images
#     A[:,i]= (A[:,i].-minimum(A[:,i]))./maximum(A[:,i])
# end

# matrice_img=affichage(A,10,19,19,1)
# file_name="CBCL_person.png"
# save(file_name,matrice_img)

indices_melanges = shuffle(1:size(A, 2))
Person=A[:,indices_melanges]
W_true=W_true[indices_melanges,:]
# matrice_img=affichage(Person,10,19,19,1)
# file_name="CBCL_100.png"
# save(file_name,matrice_img)

X=Person'*Person
###########OPTIONS##################################################
init="sspa"
maxiter=10000
timelimit=5
epsi=10e-7
erreur=zeros(3)
accu=zeros(3)
temps_execution=zeros(3)
temps_execution[1] = @elapsed begin
    W, S, erreur[1] = symTriONMF_coordinate_descent(X, r, maxiter, epsi,init, timelimit)
end
println("1")
#accu[1]=calcul_accuracy(W_true,W)
temps_execution[2] = @elapsed begin
    W2, S2, erreur[2] = symTriONMF_update_rules(X, r, maxiter, epsi,init)
end
#accu[2]=calcul_accuracy(W_true,W2)
println("2")
temps_execution[3] = @elapsed begin
    W3, H, erreur[3] = alternatingONMF(X, r, maxiter, epsi,init)
end
#accu[3]=calcul_accuracy(W_true,H')
method=["CD","MU","ONMF"]
for i in 1:3
    println(method[i])
    println("temps :",temps_execution[i]," s")
    println("erreur : ",erreur[i])
    println("accuracy: ",accu[i])
end


# classement=Vector{Vector{Int}}(undef, r)  # Initialisation d'un vecteur de r éléments
# global vide=0
# for i=1:r
    
#     classement[i] = findall(W[:, i] .> 0)
#     if size(classement[i])[1]>0
#         matrice_img=affichage(Person[:,classement[i]],size(classement[i])[1],19,19,1)
#         file_name="CBCL_$i.png"
#         save(file_name,matrice_img)
#     else
#         global vide=vide+1
#         println("classe vide ")   
#     end          
# end 
# println(vide)

# matrice=S./norm(S,2)
# for i in 1:r
#     matrice[i,i]=0
# end
# # Obtenir la palette de couleurs en niveaux de gris
# grays = cgrad(:grays)

# # Inverser l'ordre des couleurs pour obtenir des niveaux de gris inversés
# inverted_grays = reverse(grays)

# # Créer l'objet carte thermique avec la palette de couleurs inversée
# heatmap_obj = heatmap(matrice, color=inverted_grays)

# # Afficher la carte thermique
# plot(heatmap_obj)

