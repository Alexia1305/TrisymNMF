using MAT
using LinearAlgebra
using Plots
function affiche(X,file_name)
        r,_=size(X)
        # Afficher la heatmap
        grayscale_palette = cgrad([:white, :black])
        heatmap(X, color=grayscale_palette,
        xticks=(1:r, 1:r), yticks=(1:r, 1:r))
        savefig(file_name)
end 