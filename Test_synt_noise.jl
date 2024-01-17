include("algo_symTriONMF.jl")
using DelimitedFiles
using Plots
gr()  # Configurer le backend GR
using SparseArrays
nbr_level=5
n=200
const rin=8
epsilon=collect(range(0, stop=1, length=nbr_level))
result=zeros(length(epsilon),4)
result2=zeros(length(epsilon), 4)
for level in 1:nbr_level
    r=rin
    println(level)
    nbr_test=200
    accuracy_moy=0
    accuracy_moy2=0
    time1=0
    time2=0
    error1=0
    error2=0
    succes=0
    succes2=0
    for test in 1:nbr_test
        println(test)
        
        W_true2 = zeros(n, r)
        for i in 1:n
            k = rand(1:r)
            W_true2[i, k] = rand()+1
        end
        
        # Supprimer les colonnes nulles
        # Trouver les indices des colonnes non-nulles
        indices_colonnes_non_nulles = findall(x -> any(x .!= 0), eachcol(W_true2))

        # Extraire les colonnes non-nulles
        W_true = W_true2[:, indices_colonnes_non_nulles]
        r = size(W_true, 2)
        #normaliser 
        for j in 1:r
            W_true2[:, j] .= W_true2[:, j] ./ norm(W_true2[:, j],2)
        end
        # Densité de la matrice (proportion d'éléments non nuls)
        density = 0.3
        
        # Générer une matrice sparse aléatoire
        random_sparse_matrix = sprand(r, r, density)
        S=Matrix(random_sparse_matrix)
        S = 0.5 * (S + transpose(S))
        # Mettre les éléments diagonaux à 1
        for k in 1:r
            S[k,k]=1
        end 
        X = W_true * S* transpose(W_true)

        #ajout du bruit 
        if epsilon[level] != 0 
            N = randn(n,n); 
            N = epsilon[level] * (N/norm(N))*norm(X);  
            X=X+N
            X=max.(X, 0) # pas de vaelurs négatives 
        end 
        maxiter=10000000
        epsi=10e-5
        # algorithme :
        temps_execution_1 = @elapsed begin
            W, S, erreur = symTriONMF_coordinate_descent(X, r, maxiter,epsi,true)
        end
        temps_execution_2 = @elapsed begin
            W2, S2, erreur2 = symTriONMF_update_rules(X, r, maxiter,epsi,true)
        end 
        accu1=calcul_accuracy(W_true,W)
        accu2=calcul_accuracy(W_true,W2)
        accuracy_moy += accu1
        accuracy_moy2 += accu2
        if accu1==1
            succes+=1
        end
        if accu2==1
            succes2+=1
        end
        time1+=temps_execution_1
        time2+=temps_execution_2
        error1+=erreur
        error2+=erreur2
        
        
        
    end 
    accuracy_moy/=nbr_test
    accuracy_moy2/=nbr_test
    time1/=nbr_test
    time2/=nbr_test
    error1/=nbr_test
    error2/=nbr_test
    println(accuracy_moy)
    println(accuracy_moy2)
    result[level,1]=accuracy_moy
    result2[level,1]=accuracy_moy2
    result[level,2]=time1
    result2[level,2]=time2
    result[level,3]=error1
    result2[level,3]=error2
    result[level,4]=succes/nbr_test
    result2[level,4]=succes2/nbr_test

end 
plot(epsilon, result[:,4],label="coordinate_descent ", xlabel="n", ylabel="Success rate", title="Evolution of the success rate as a function of n and r",ylim=(0,1),linecolor=:blue)
scatter!(epsilon, result[:,4],label="",markercolor=:blue)
plot!(epsilon, result2[:,4],label="multiplicative updates",linestyle=:dash,linecolor=:red)
scatter!(epsilon, result2[:,4],label="",markercolor=:red)


# Enregistrer la figure au format PNG (vous pouvez utiliser d'autres formats comme SVG, PDF, etc.)
savefig("figure4.png")
plot(epsilon, result[:,1],label="coordinate_descent", xlabel="epsilon", ylabel="accuracy", title="Evolution of the accuracy a function of epsilon",ylim=(0.5, 1),linecolor=:blue)
scatter!(epsilon, result[:,1],label="",markercolor=:blue)
plot!(epsilon, result2[:,1],label="multiplicative updates",linecolor=:red)
scatter!(epsilon, result2[:,1],label="",markercolor=:red)

# Enregistrer la figure au format PNG (vous pouvez utiliser d'autres formats comme SVG, PDF, etc.)
savefig("figure.png")

plot(epsilon, result[:,2],label="coordinate_descent", xlabel="epsilon", ylabel="time [s]", title="Evolution of the resolution time a function of epsilon",linecolor=:blue)
scatter!(epsilon, result[:,2],label="",markercolor=:blue)
plot!(epsilon, result2[:,2],label="multiplicative updates",linecolor=:red)
scatter!(epsilon, result2[:,2],label="",markercolor=:red)

# Enregistrer la figure au format PNG (vous pouvez utiliser d'autres formats comme SVG, PDF, etc.)
savefig("figure2.png")


plot(epsilon, result[:,3],label="coordinate_descent", xlabel="epsilon", ylabel="relative error", title="Evolution of the relative error as a function of epsilon",ylim=(0,:auto),linecolor=:blue)
scatter!(epsilon, result[:,3],label="",markercolor=:blue)
plot!(epsilon, result2[:,3],label="multiplicative updates",linecolor=:red)
scatter!(epsilon, result2[:,3],label="",markercolor=:red)

# Enregistrer la figure au format PNG (vous pouvez utiliser d'autres formats comme SVG, PDF, etc.)
savefig("figure3.png")

# Nom du fichier
nom_fichier = "noise_kmeans.txt"

# Écriture des données dans le fichier
writedlm(nom_fichier, [epsilon result result2], ',')