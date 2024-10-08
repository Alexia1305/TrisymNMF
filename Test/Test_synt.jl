include("../algo/OtrisymNMF.jl")
using DelimitedFiles
using Plots
using LaTeXStrings

gr()  # Configurer le backend GR
using SparseArrays
using Random
Random.seed!(2001)
dim_n=[10,30,50,100,150,200]
dim_r=[2,3,4,5,6,7]
nbr_algo=4
result=zeros(length(dim_n), 4)
result2=zeros(length(dim_n),4)
result3=zeros(length(dim_n),4)
result4=zeros(length(dim_n),4)
for dim in 1:length(dim_n)
    println(dim)
    nbr_test=100
    accuracy_moy=zeros(4)
    succes=zeros(4)
    time=zeros(4)
    error=zeros(4)
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
            W, S, erreur = OtrisymNMF_CD(X, r, maxiter,epsi,"random")
        end
        temps_execution_2 = @elapsed begin
            W2, S2, erreur2 = OtrisymNMF_CD(X, r, maxiter,epsi,"k_means")
        end 
        if nbr_algo>=3
            temps_execution_3 = @elapsed begin
                W3, S3, erreur3 = OtrisymNMF_CD(X, r, maxiter,epsi,"sspa")
            end 
        end 
        temps_execution_4 = @elapsed begin
            W4, S4, erreur4 = OtrisymNMF_CD(X, r, maxiter,epsi,"spa")
        end 
        accu1=calcul_accuracy(W_true,W)
        accu2=calcul_accuracy(W_true,W2)
        
        accuracy_moy[1]+= accu1
        accuracy_moy[2] += accu2
        
        if accu1==1
            succes[1]+=1
        end
        if accu2==1
            succes[2]+=1
        end
       

        time[1]+=temps_execution_1
        time[2]+=temps_execution_2
       
        error[1]+=erreur
        error[2]+=erreur2
        
        
        if nbr_algo>=4
            accu3=calcul_accuracy(W_true,W3)
            accuracy_moy[3]+= accu3
            if accu3==1
                succes[3]+=1
            end
            time[3]+=temps_execution_3
            error[3]+=erreur3
            accu4=calcul_accuracy(W_true,W4)
            accuracy_moy[4]+= accu4
            if accu4==1
                succes[4]+=1
            end
            time[4]+=temps_execution_4
            error[4]+=erreur4
        end 
    end 
    accuracy_moy/=nbr_test 
    time/=nbr_test
    error/=nbr_test
    println(accuracy_moy)
    
    result[dim,1]=accuracy_moy[1]
    result2[dim,1]=accuracy_moy[2]
    result3[dim,1]=accuracy_moy[3]
    result4[dim,1]=accuracy_moy[4]
    result[dim,2]=time[1]
    result2[dim,2]=time[2]
    result3[dim,2]=time[3]
    result4[dim,2]=time[4]
    result[dim,3]=error[1]
    result2[dim,3]=error[2]
    result3[dim,3]=error[3]
    result4[dim,3]=error[4]
    result[dim,4]=succes[1]/nbr_test
    result2[dim,4]=succes[2]/nbr_test
    result3[dim,4]=succes[3]/nbr_test
    result4[dim,4]=succes[4]/nbr_test

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



plot(dim_n, result2[:,4],label="coordinate_descent init kmeans",ylim=(0,1),linestyle=:dash,linecolor=:red,xtickfont=font_size, ytickfont=font_size, legendfont=font_size)
scatter!(dim_n, result2[:,4],label="",markercolor=:red)
plot!(dim_n, result[:,4],label="coordinate_descent init random", xlabel="n", ylabel="Success rate",ylim=(0,1),linecolor=:blue)
scatter!(dim_n, result[:,4],label="",markercolor=:blue)
if nbr_algo >=3
    plot!(dim_n, result3[:,4],label="coordinate_descent init sspa",linestyle=:dashdot,linecolor=:green,xtickfont=font_size, ytickfont=font_size, legendfont=font_size)
    scatter!(dim_n, result3[:,4],label="",markercolor=:green)
    plot!(dim_n, result4[:,4],label="coordinate_descent init spa",linestyle=:dot,linecolor=:purple,xtickfont=font_size, ytickfont=font_size, legendfont=font_size)
    scatter!(dim_n, result4[:,4],label="",markercolor=:purple)
end 
# Enregistrer la figure au format PNG (vous pouvez utiliser d'autres formats comme SVG, PDF, etc.)
savefig("figure4.png")


plot(dim_n, result2[:,1],label="coordinate_descent init kmeans",xtickfont=font_size, ytickfont=font_size, legendfont=font_size,ylim=(0,1),linestyle=:dash,linecolor=:red)
scatter!(dim_n, result2[:,1],label="",markercolor=:red)
plot!(dim_n, result[:,1],label="coordinate_descent init random", xlabel="n", ylabel="accuracy mean", xtickfont=font_size, ytickfont=font_size, legendfont=font_size,ylim=(0,1),linecolor=:blue)
scatter!(dim_n, result[:,1],label="",markercolor=:blue)
if nbr_algo >=3
    plot!(dim_n, result3[:,1],label="coordinate_descent init sspa",linestyle=:dashdot,linecolor=:green)
    scatter!(dim_n, result3[:,1],label="",markercolor=:green)
    plot!(dim_n, result4[:,1],label="coordinate_descent init spa",linestyle=:dot,linecolor=:purple)
    scatter!(dim_n, result4[:,1],label="",markercolor=:purple)
end 

# Enregistrer la figure au format PNG (vous pouvez utiliser d'autres formats comme SVG, PDF, etc.)
savefig("figure.png")


plot(dim_n, result2[:,2],label="coordinate_descent init kmeans",xtickfont=font_size, ytickfont=font_size, legendfont=font_size,linestyle=:dash,linecolor=:red)
scatter!(dim_n, result2[:,2],label="",markercolor=:red)
plot!(dim_n, result[:,2],label="coordinate_descent init random ", xlabel="n", ylabel="time [s]", xtickfont=font_size, ytickfont=font_size, legendfont=font_size,linecolor=:blue)
scatter!(dim_n, result[:,2],label="",markercolor=:blue)
if nbr_algo >=3
    plot!(dim_n, result3[:,2],label="coordinate_descent init sspa",linestyle=:dashdot,linecolor=:green)
    scatter!(dim_n, result3[:,2],label="",markercolor=:green)
    plot!(dim_n, result4[:,2],label="coordinate_descent init spa",linestyle=:dot,linecolor=:purple)
    scatter!(dim_n, result4[:,2],label="",markercolor=:purple)
end 

# Enregistrer la figure au format PNG (vous pouvez utiliser d'autres formats comme SVG, PDF, etc.)
savefig("figure2.png")


plot(dim_n, result2[:,3],label="coordinate_descent init kmeans",xlabel="n", ylabel="relative error", title=xtickfont=font_size, ytickfont=font_size, legendfont=font_size,ylim=(0,:auto),linestyle=:dash,linecolor=:red)
scatter!(dim_n, result2[:,3],label="",markercolor=:red)
plot!(dim_n, result[:,3],label="coordinate_descent init random", xlabel="n", ylabel="relative error", title=xtickfont=font_size, ytickfont=font_size, legendfont=font_size,ylim=(0,:auto),linecolor=:blue)
scatter!(dim_n, result[:,3],label="",markercolor=:blue)
if nbr_algo >=3
    plot!(dim_n, result3[:,3],label="coordinate_descent init sspa",linestyle=:dashdot,linecolor=:green)
    scatter!(dim_n, result3[:,3],label="",markercolor=:green)
    plot!(dim_n, result4[:,3],label="coordinate_descent init spa",linestyle=:dot,linecolor=:purple)
    scatter!(dim_n, result4[:,3],label="",markercolor=:purple)
end 
# Enregistrer la figure au format PNG (vous pouvez utiliser d'autres formats comme SVG, PDF, etc.)
savefig("figure3.png")

# Nom du fichier
nom_fichier = "random_and_kmeans_and_spa.txt"

# Écriture des données dans le fichier
writedlm(nom_fichier, [dim_n dim_r result result2 result3], ',')