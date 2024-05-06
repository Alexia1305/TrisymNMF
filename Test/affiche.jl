using MAT
using LinearAlgebra
using Plots
mat = matread("WSwimmer.mat")
X = mat["S"]
for i in 1:18
    X[i,i]=0
end

# Afficher la heatmap
grayscale_palette = cgrad([:white, :black])
heatmap(X, color=grayscale_palette,
        xticks=(1:18, 1:18), yticks=(1:18, 1:18))
savefig("heatmap_swim.png")
