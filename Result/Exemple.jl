include("algo_symTriONMF.jl")
using DelimitedFiles
using Plots
gr()  # Configurer le backend GR
using SparseArrays
# création matrice
n = 50
r = 10
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

temps_execution_2 = @elapsed begin
    W2, S2, erreur2 = symTriONMF_update_rules(X, r, maxiter,epsi,true)
end 


println(W2)
println(transpose(W2)*W2)