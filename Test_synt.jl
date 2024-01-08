include("algo_symTriONMF.jl")
using Plots
gr()  # Configurer le backend GR
using SparseArrays
dim_n=[10,20,50,100,150,200]
dim_r=[2,3,4,5,6,7]
result=zeros(length(dim_n), 3)
result2=zeros(length(dim_n), 3)
resultrdm=zeros(length(dim_n), 3)
result2rdm=zeros(length(dim_n), 3)
for dim in 1:length(dim_n)
    println(dim)
    nbr_test=100
    accuracy_moy=0
    accuracy_moy2=0
    time1=0
    time2=0
    error1=0
    error2=0
    accuracy_moyrdm=0
    accuracy_moy2rdm=0
    time1rdm=0
    time2rdm=0
    error1rdm=0
    error2rdm=0
    for test in 1:nbr_test
        # création matrice
        n = dim_n[dim]
        r = dim_r[dim]
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
        density = 0.4
        
        # Générer une matrice sparse aléatoire
        random_sparse_matrix = sprand(r, r, density)
        S=Matrix(random_sparse_matrix)
        S = 0.5 * (S + transpose(S))
        # Mettre les éléments diagonaux à 1
        for k in 1:r
            S[k,k]=1
        end 
        X = W_true * S* transpose(W_true)
        maxiter=1000
        epsi=10e-5
        # algorithme :
        temps_execution_1 = @elapsed begin
            W, S, erreur = symTriONMF_coordinate_descent(X, r, maxiter,epsi,true)
        end
        temps_execution_2 = @elapsed begin
            W2, S2, erreur2 = symTriONMF_update_rules(X, r, maxiter,epsi,true)
        end 
        temps_execution_1rdm = @elapsed begin
            Wrdm, S, erreurrdm = symTriONMF_coordinate_descent(X, r, maxiter,epsi,false)
        end
        temps_execution_2rdm = @elapsed begin
            W2rdm, S2, erreur2rdm = symTriONMF_update_rules(X, r, maxiter,epsi,false)
        end 
        accuracy_moy += calcul_accuracy(W_true,W)
        accuracy_moy2 += calcul_accuracy(W_true,W2)
        time1+=temps_execution_1
        time2+=temps_execution_2
        error1+=erreur
        error2+=erreur2
        accuracy_moyrdm += calcul_accuracy(W_true,W)
        accuracy_moy2rdm += calcul_accuracy(W_true,W2)
        time1rdm+=temps_execution_1rdm
        time2rdm+=temps_execution_2rdm
        error1rdm+=erreurrdm
        error2rdm+=erreur2rdm
        
        
    end 
    accuracy_moy/=nbr_test
    accuracy_moy2/=nbr_test
    time1/=nbr_test
    time2/=nbr_test
    error1/=nbr_test
    error2/=nbr_test
    accuracy_moyrdm/=nbr_test
    accuracy_moy2rdm/=nbr_test
    time1rdm/=nbr_test
    time2rdm/=nbr_test
    error1rdm/=nbr_test
    error2rdm/=nbr_test
    println(accuracy_moy)
    println(accuracy_moy2)
    result[dim,1]=accuracy_moy
    result2[dim,1]=accuracy_moy2
    result[dim,2]=time1
    result2[dim,2]=time2
    result[dim,3]=error1
    result2[dim,3]=error2
    resultrdm[dim,1]=accuracy_moyrdm
    result2rdm[dim,1]=accuracy_moy2rdm
    resultrdm[dim,2]=time1rdm
    result2rdm[dim,2]=time2rdm
    resultrdm[dim,3]=error1rdm
    result2rdm[dim,3]=error2rdm

end 
plot(dim_n, result[:,1],label="coordinate_descent init kmeans", xlabel="n", ylabel="accuracy", title="Evolution of the accuracy a function of n and r",ylim=(0, :auto))
scatter!(dim_n, result[:,1],label="coordinate_descent",legend=false)
plot!(dim_n, result2[:,1],label="update_rules init kmeans")
scatter!(dim_n, result2[:,1],label="update_rules",legend=false)
plot(dim_n, resultrdm[:,1],label="coordinate_descent init random", xlabel="n", ylabel="accuracy", title="Evolution of the accuracy a function of n and r",ylim=(0, :auto))
scatter!(dim_n, resultrdm[:,1],label="coordinate_descent",legend=false)
plot!(dim_n, result2rdm[:,1],label="update_rules init random")
scatter!(dim_n, result2rdm[:,1],label="update_rules",legend=false)
# Enregistrer la figure au format PNG (vous pouvez utiliser d'autres formats comme SVG, PDF, etc.)
savefig("figure.png")

plot(dim_n, result[:,2],label="coordinate_descent init kmeans", xlabel="n", ylabel="time [s]", title="Evolution of the resolution time a function of n and r")
scatter!(dim_n, result[:,2],label="coordinate_descent",legend=false)
plot!(dim_n, result2[:,2],label="update_rules init kmeans")
scatter!(dim_n, result2[:,2],label="update_rules",legend=false)
plot(dim_n, resultrdm[:,2],label="coordinate_descent init random", xlabel="n", ylabel="time [s]", title="Evolution of the resolution time a function of n and r")
scatter!(dim_n, resultrdm[:,2],label="coordinate_descent",legend=false)
plot!(dim_n, result2rdm[:,2],label="update_rules init random")
scatter!(dim_n, result2rdm[:,2],label="update_rules",legend=false)
# Enregistrer la figure au format PNG (vous pouvez utiliser d'autres formats comme SVG, PDF, etc.)
savefig("figure2.png")


plot(dim_n, result[:,3],label="coordinate_descent init kmeans", xlabel="n", ylabel="relative error", title="Evolution of the relative error as a function of n and r",ylim=(0,:auto))
scatter!(dim_n, result[:,3],label="coordinate_descent",legend=false)
plot!(dim_n, result2[:,3],label="update_rules init kmeans ")
scatter!(dim_n, result2[:,3],label="update_rules",legend=false)
plot(dim_n, resultrdm[:,3],label="coordinate_descent init random", xlabel="n", ylabel="relative error", title="Evolution of the relative error as a function of n and r",ylim=(0,:auto))
scatter!(dim_n, resultrdm[:,3],label="coordinate_descent",legend=false)
plot!(dim_n, result2rdm[:,3],label="update_rules init random")
scatter!(dim_n, result2rdm[:,3],label="update_rules",legend=false)

# Enregistrer la figure au format PNG (vous pouvez utiliser d'autres formats comme SVG, PDF, etc.)
savefig("figure3.png")