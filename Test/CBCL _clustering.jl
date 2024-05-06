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
function erreur_kmeans(X, assignments, centroids)
    n = size(X, 1)  # Nombre total de points
    r = length(centroids)  # Nombre total de clusters
    
    # Initialiser l'erreur
    erreur = 0.0
    
    # Calculer la somme des distances au carré de chaque point à son centroïde attribué
    for i in 1:n
        cluster_index = assignments[i]  # Indice du cluster attribué au point i
        centroid = centroids[:,cluster_index]  # Centroïde associé au cluster
        distance = Euclidean()(X[i, :], centroid)  # Distance entre le point et son centroïde
        erreur += distance^2
    end
    
    return erreur
end


function clustering()

    Random.seed!(123)
    file_path = "dataset/CBCL.mat"
    mat = matread(file_path)
    A = mat["X"]
    nbr_algo=4
    nbr_test=50
    groupes=[2,5,10,20,30,40,50]
    nbr_groupe=length(groupes)


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
        collect(49:53),
        collect(54:56),
        collect(57:63),
        collect(64:72),
        collect(73:77),
        collect(78:81),
        collect(82:87),
        collect(88:93),
        collect(94:97),
        collect(98:103),
        collect(104:107),
        collect(108:109),
        collect(110:112),
        collect(113:117),
        collect(118:125),
        collect(126:131),
        collect(132:137),
        collect(138:140),
        collect(141:146),
        collect(147:150),
        collect(151:159),
        collect(157:162),
        collect(163:168),
        collect(169:174),
        collect(175:178),
        collect(179:183),
        collect(184:189),
        collect(190:195),
        collect(196:199),
        collect(200:203),
        collect(204:209),
        collect(210:215),
        collect(216:220),
        collect(221:226),
        collect(227:233),
        collect(234:241),
        collect(242:245),
        collect(246:250),
        collect(251:256),
        collect(257:261),
        collect(262:267)
    ]
    erreur_moy=zeros(nbr_groupe,nbr_algo)
       
    temps_execution=zeros(nbr_groupe,nbr_algo)

    for (indice, r) in enumerate(groupes)
        label_true=[]
        label_CD=[]
        label2_MU=[]
        label3_ONMF=[]
        label4_K=[]
        

        for test in 1:nbr_test
            # selection de r groupes alétoirement
            indices_aleatoires = randperm(length(groupe))[1:r]
            groupe_select=groupe[indices_aleatoires]
            picture_select= [item for sublist in groupe_select for item in sublist]
            Ag=A[:,picture_select]
            
            label=[]
            last=0
            for g in groupe_select
                nvx_groupe=last.+collect(1:length(g))
                push!(label,nvx_groupe)
                last+=length(g)
            end
            
            W_true=zeros(length(picture_select),r)
            for (indice, element) in enumerate(label)
                W_true[element,indice] .= 1
            end 


            # matrice_img=affichage(Ag,10,19,19,1)
            # file_name="CBCL_person_$test.png"
            # save(file_name,matrice_img)

            # # prepocessing
            # for i in 1: nbr_images
            #     A[:,i]= (A[:,i].-minimum(A[:,i]))./maximum(A[:,i])
            # end


            indices_melanges = shuffle(1:size(Ag, 2))
            Person=Ag[:,indices_melanges]
            W_true=W_true[indices_melanges,:]
            push!(label_true,[findfirst(x -> x != 0, ligne) for ligne in eachrow(W_true)])
            # matrice_img=affichage(Person,10,19,19,1)
            # file_name="CBCL_person_$test.png"
            # save(file_name,matrice_img)

            X=Person'*Person

            ###########OPTIONS##################################################
            init="sspa"
            maxiter=100000
            timelimit=40
            epsi=10e-7

            temps_execution[indice,1] += @elapsed begin
                W, S, erreur= symTriONMF_coordinate_descent(X, r, maxiter, epsi,init, timelimit)
            end
            erreur_moy[indice,1] +=erreur
            println("1")
            push!(label_CD,[findfirst(x -> x != 0, ligne) for ligne in eachrow(W)])
            
            temps_execution[indice,2] += @elapsed begin
                W2, S2, erreur = symTriONMF_update_rules(X, r, maxiter, epsi,init)
            end
            erreur_moy[indice,2] +=erreur
            println("2")
            push!(label2_MU,[findfirst(x -> x != 0, ligne) for ligne in eachrow(W2)])

            temps_execution[indice,3] += @elapsed begin
                W3, H, erreur= alternatingONMF(X, r, maxiter, epsi,init)
            end
            erreur_moy[indice,3] +=erreur
            push!(label3_ONMF,[findfirst(x -> x != 0, ligne) for ligne in eachrow(H')])

            temps_execution[indice,4] += @elapsed begin
                n = size(X, 1)
                Xnorm = similar(X, Float64)
                # Pour chaque colonne de X
                for i in 1:n
                    # Calcul de la norme euclidienne de la colonne
                    col_norm = norm(X[:, i],2)
                    # Division de la colonne par sa norme
                    if col_norm !=0
                        Xnorm[:, i] = X[:, i] / col_norm
                    else
                        Xnorm[:, i] = X[:, i] 
                end 
            end
                R=kmeans(Xnorm,r,maxiter=Int(ceil(1000)))
            end
            push!(label4_K, assignments(R))
           
            
            centroids = R.centers

            # Calculer l'erreur K-means
            erreur_moy[indice,4] += erreur_kmeans(Xnorm,assignments(R), centroids)
        


            if r<=5
                println(calcul_accuracy(W_true,W))
                println(calcul_accuracy(W_true,W2))
                println(calcul_accuracy(W_true,H'))
            end

        end 

        erreur_moy[indice,:]=erreur_moy[indice,:]./nbr_test
        temps_execution[indice,:]=temps_execution[indice,:]./nbr_test

        ####### ECRITURE EN MATLAB POur calculer l'accuracy #########
       

        
        # Définir le nom du fichier .mat (sans le chemin)
        nom_fichier = "CBCL_cluster_$r.mat"

        # Définir le chemin du dossier parent où se trouve le dossier "Test"
        chemin_parent = joinpath(@__DIR__, "..")  # ".." indique le dossier parent

        # Définir le chemin du dossier "Calcul_acc" dans le dossier parent "Test"
        dossier = joinpath(chemin_parent, "Test", "Calcul_acc")

        # Concaténer le chemin du dossier et le nom du fichier pour obtenir le chemin complet
        chemin_fichier = joinpath(dossier, nom_fichier)
        # Écrire les données dans un fichier .mat
        matfile = matopen(chemin_fichier, "w")
        write(matfile, "label_true", label_true)
        write(matfile, "label_CD", label_CD)
        write(matfile, "label4_K", label4_K)
        write(matfile, "label3_ONMF", label3_ONMF)
        write(matfile, "label2_MU", label2_MU)
        close(matfile)  # Assurez-vous de fermer le fichier après l'écriture des données


        println(r)
        println(erreur_moy[indice,:])
        println(temps_execution[indice,:])

    end 
    
# Définir le chemin du fichier texte de sortie
chemin_fichier = "CBCL_cluster.txt"

# Ouvrir le fichier en mode écriture
fichier = open(chemin_fichier, "w")

# Écrire les données dans le fichier texte

write(fichier, "r: ",string(groupes),"\n")
write(fichier, "Temps d'exécution : ", string(temps_execution), " secondes\n")
write(fichier, "Erreur : ", string(erreur_moy))

# Fermer le fichier
close(fichier)
   
end 

# clustering()

groupe=[
       
        collect(18:20),
        collect(200:203),
        collect(41:44),
        collect(94:97),
        collect(49:53)
    ]
picture_select= [item for sublist in groupe for item in sublist]
Ag=A[:,picture_select]
indices_melanges = shuffle(1:size(Ag, 2))
Person=Ag[:,indices_melanges]

matrice_img=affichage(Person,5,19,19,1)
file_name="CBCL_5.png"
save(file_name,matrice_img)
# file_path = "dataset/CBCL.mat"
# mat = matread(file_path)
# A = mat["X"]
# matrice_img=affichage(A[:,241:270],10,19,19,1)





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

