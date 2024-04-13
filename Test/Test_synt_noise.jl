include("../algo/algo_symTriONMF.jl")
using DelimitedFiles
using Plots
gr()  # Configurer le backend GR
using SparseArrays
using LaTeXStrings
nbr_level=5
n=200
const rin=8
epsilon=collect(range(0, stop=1, length=nbr_level))
result=zeros(length(epsilon),4)
result2=zeros(length(epsilon), 4)
result3=zeros(length(epsilon), 4)
result4=zeros(length(epsilon), 4)
for level in 1:nbr_level
    r=rin
    println(level)
    nbr_test=100
    accuracy_moy=0
    accuracy_moy2=0
    time1=0
    time2=0
    error1=0
    error2=0
    succes=0
    succes2=0
    accuracy_moy3=0
    time3=0
    error3=0
    succes3=0
    accuracy_moy4=0
    time4=0
    error4=0
    succes4=0
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
        maxiter=10000
        epsi=10e-5
        # algorithme :
        temps_execution_1 = @elapsed begin
            W, S, erreur = symTriONMF_coordinate_descent(X, r, maxiter,epsi,"k_means")
        end
        temps_execution_2 = @elapsed begin
            W2, S2, erreur2 = symTriONMF_coordinate_descent(X, r, maxiter,epsi,"sspa")
        end
        temps_execution_3 = @elapsed begin
            W3, S3, erreur3 = symTriONMF_coordinate_descent(X, r, maxiter,epsi,"random")
        end
        temps_execution_4 = @elapsed begin
            W4, S4, erreur4 = symTriONMF_coordinate_descent(X, r, maxiter,epsi,"spa")
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
        accu3=calcul_accuracy(W_true,W3)
        accuracy_moy3 += accu3
        if accu3==1
            succes3+=1
        end
        time3+=temps_execution_3
        error3+=erreur3
        accu4=calcul_accuracy(W_true,W4)
        accuracy_moy4 += accu4
        if accu4==1
            succes4+=1
        end
        time4+=temps_execution_4
        error4+=erreur4
    end 
    accuracy_moy/=nbr_test
    accuracy_moy2/=nbr_test
    time1/=nbr_test
    time2/=nbr_test
    error1/=nbr_test
    error2/=nbr_test
    println(accuracy_moy)
    println(accuracy_moy2)
    println(accuracy_moy3/nbr_test)
    println(accuracy_moy4/nbr_test)
    result[level,1]=accuracy_moy
    result2[level,1]=accuracy_moy2
    result[level,2]=time1
    result2[level,2]=time2
    result[level,3]=error1
    result2[level,3]=error2
    result[level,4]=succes/nbr_test
    result2[level,4]=succes2/nbr_test
    
    result3[level,1]=accuracy_moy3/nbr_test
    result3[level,2]=time3/nbr_test
    result3[level,3]=error3/nbr_test
    result3[level,4]=succes3/nbr_test
    result4[level,1]=accuracy_moy4/nbr_test
    result4[level,2]=time4/nbr_test
    result4[level,3]=error4/nbr_test
    result4[level,4]=succes4/nbr_test
end 
# Taille de la police
font_size = 11
plot_font = "Computer Modern"
default(
    fontfamily=plot_font,
    guidefontsize=font_size,
    linewidth=2, 
    framestyle=:box, 
    label=nothing, 
    grid=false
)
plot(epsilon, result[:,4],label="init kmeans", xlabel="n", ylabel="Success rate", xtickfont=font_size, ytickfont=font_size, legendfont=font_size,linecolor=:blue)
scatter!(epsilon, result[:,4],label="",markercolor=:blue)


plot!(epsilon, result2[:,4],label="init sspa",linestyle=:dash,linecolor=:red)
scatter!(epsilon, result2[:,4],label="",markercolor=:red)
plot!(epsilon, result4[:,4],label="init spa",linestyle=:dot,linecolor=:purple)
scatter!(epsilon, result4[:,4],label="",markercolor=:purple)
plot!(epsilon, result3[:,4],label="init random",linestyle=:dashdot,linecolor=:green)
scatter!(epsilon, result3[:,4],label="",markercolor=:green)


# Enregistrer la figure au format PNG (vous pouvez utiliser d'autres formats comme SVG, PDF, etc.)
savefig("figure4.png")
plot(epsilon, result[:,1],label="init kmeans", xlabel="epsilon", ylabel="accuracy", xtickfont=font_size, ytickfont=font_size, legendfont=font_size,linecolor=:blue)
scatter!(epsilon, result[:,1],label="",markercolor=:blue)

plot!(epsilon, result2[:,1],label="init sspa",linecolor=:red,linestyle=:dash)
scatter!(epsilon, result2[:,1],label="",markercolor=:red)
plot!(epsilon, result4[:,1],label="init spa",linestyle=:dot,linecolor=:purple)
scatter!(epsilon, result4[:,1],label="",markercolor=:purple)
plot!(epsilon, result3[:,1],label="init random",linestyle=:dashdot,linecolor=:green)
scatter!(epsilon, result3[:,1],label="",markercolor=:green)
# Enregistrer la figure au format PNG (vous pouvez utiliser d'autres formats comme SVG, PDF, etc.)
savefig("figure.png")

plot(epsilon, result[:,2],label="init kmeans", xlabel="epsilon", ylabel="time [s]", xtickfont=font_size, ytickfont=font_size, legendfont=font_size,linecolor=:blue)
scatter!(epsilon, result[:,2],label="",markercolor=:blue)

plot!(epsilon, result2[:,2],label="init sspa",linecolor=:red,linestyle=:dash)
scatter!(epsilon, result2[:,2],label="",markercolor=:red)
plot!(epsilon, result4[:,2],label="init spa",linestyle=:dot,linecolor=:purple)
scatter!(epsilon, result4[:,2],label="",markercolor=:purple)
plot!(epsilon, result3[:,2],label="init random",linestyle=:dashdot,linecolor=:green)
scatter!(epsilon, result3[:,2],label="",markercolor=:green)

# Enregistrer la figure au format PNG (vous pouvez utiliser d'autres formats comme SVG, PDF, etc.)
savefig("figure2.png")


plot(epsilon, result[:,3],label="init kmeans", xlabel="epsilon", ylabel="relative error", xtickfont=font_size, ytickfont=font_size, legendfont=font_size,ylim=(0,:auto),linecolor=:blue)
scatter!(epsilon, result[:,3],label="",markercolor=:blue)

plot!(epsilon, result2[:,3],label="init sspa",linecolor=:red,linestyle=:dash)
scatter!(epsilon, result2[:,3],label="",markercolor=:red)
plot!(epsilon, result4[:,3],label="init spa",linestyle=:dot,linecolor=:purple)
scatter!(epsilon, result4[:,3],label="",markercolor=:purple)
plot!(epsilon, result3[:,3],label="init random",linestyle=:dashdot,linecolor=:green)
scatter!(epsilon, result3[:,3],label="",markercolor=:green)

# Enregistrer la figure au format PNG (vous pouvez utiliser d'autres formats comme SVG, PDF, etc.)
savefig("figure3.png")

# Nom du fichier
nom_fichier = "noise_kmeans_sSpa.txt"

# Écriture des données dans le fichier
writedlm(nom_fichier, [epsilon result result2], ',')