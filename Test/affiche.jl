using MAT
using LinearAlgebra
using Plots
mat = matread("St.mat")
X = mat["S"]
println(X)

# Afficher la heatmap
grayscale_palette = cgrad([:white, :black])
heatmap(X, color=grayscale_palette,
        xticks=(1:15, 1:15), yticks=(1:15, 1:15))
savefig("heatmap_TDT2_TRi.png")
