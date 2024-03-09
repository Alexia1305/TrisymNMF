using Plots
mat = matread("WSwimmer.mat")
X = mat["S"]

matrice=X./norm(X,2)
for i in 1:5
    matrice[i,i]=0
end
X=matrice'
# Afficher la heatmap
# Afficher la heatmap
Plots.heatmap(matrice, c=:grays, xlabel="Colonnes", ylabel="Lignes", clim=(maximum(matrice), minimum(matrice)))