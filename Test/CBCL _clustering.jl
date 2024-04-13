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

nbr_data=5
# Charger le fichier karate.mat
file_path = "dataset/CBCL.mat"
mat = matread(file_path)
A = mat["X"]
erreur=zeros(nbr_data,3)
accu=zeros(nbr_data,3)
temps_execution=zeros(nbr_data,3)

for data in 1:nbr_data
    println("data : ",data)
    if  data==1
        
        nbr_images=50
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
        Ag=A[:,1:nbr_images]
    elseif data==2 
        nbr_images=50
        r=10
        groupe=[
            collect(1:3),
            collect(4:6),
            collect(7:13),
            collect(14:22),
            collect(23:27),
            collect(28:31),
            collect(32:37),
            collect(38:43),
            collect(44:47),
            collect(48:50)
        ]
        Ag=A[:,51:100]
    elseif data==3
        nbr_images=50
        r=10
        groupe=[
            collect(1:3),
            collect(4:9),
            collect(10:12),
            collect(13:17),
            collect(18:25),
            collect(26:31),
            collect(32:37),
            collect(38:40),
            collect(41:46),
            collect(47:50)
        ]
        Ag=A[:,101:150]
    elseif data==4
        r=10
        nbr_images=49
        groupe=[
            collect(1:9),
            collect(7:12),
            collect(13:18),
            collect(19:24),
            collect(25:28),
            collect(29:33),
            collect(34:39),
            collect(40:44),
            collect(45:48),
            collect(49)
        ]
        Ag=A[:,152:200]

    elseif data==5
        nbr_images=50
        r=9
        groupe=[
            [1,2,3],
            collect(4:9),
            collect(10:15),
            collect(16:20),
            collect(21:26),
            collect(27:33),
            collect(34:41),
            collect(42:45),
            collect(46:50),
            
        ]

        Ag=A[:,201:250]
    end
    W_true=zeros(nbr_images,r)
    for (indice, element) in enumerate(groupe)
        W_true[element,indice] .= 1
    end 


    # matrice_img=affichage(Ag,10,19,19,1)
    # file_name="CBCL_person_$data.png"
    # save(file_name,matrice_img)

    # # prepocessing
    # for i in 1: nbr_images
    #     A[:,i]= (A[:,i].-minimum(A[:,i]))./maximum(A[:,i])
    # end


    indices_melanges = shuffle(1:size(Ag, 2))
    Person=Ag[:,indices_melanges]
    W_true=W_true[indices_melanges,:]
    # matrice_img=affichage(Person,10,19,19,1)
    # file_name="CBCL_person_$data.png"
    # save(file_name,matrice_img)

    X=Person'*Person
    ###########OPTIONS##################################################
    init="sspa"
    maxiter=100000
    timelimit=5
    epsi=10e-7
    
    temps_execution[data,1] = @elapsed begin
        W, S, erreur[data,1] = symTriONMF_coordinate_descent(X, r, maxiter, epsi,init, timelimit)
    end
    println("1")
    accu[data,1]=calcul_accuracy(W_true,W)
    temps_execution[data,2] = @elapsed begin
        W2, S2, erreur[data,2] = symTriONMF_update_rules(X, r, maxiter, epsi,init)
    end
    accu[data,2]=calcul_accuracy(W_true,W2)
    println("2")
    temps_execution[data,3] = @elapsed begin
        W3, H, erreur[data,3] = alternatingONMF(X, r, maxiter, epsi,init)
    end
    accu[data,3]=calcul_accuracy(W_true,H')
    method=["CD","MU","ONMF"]
    for i in 1:3
        println(method[i])
        println("temps :",temps_execution[data,i]," s")
        println("erreur : ",erreur[data,i])
        println("accuracy: ",accu[data,i])
    end
end 
method=["CD","MU","ONMF"]

for i in 1:3
    println(method[i])
    println("temps moy :",sum(temps_execution[:,i])/nbr_data," s")
    println("erreur moy : ",sum(erreur[:,i])/nbr_data)
    println("accuracy moy: ",sum(accu[:,i])/nbr_data)
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

