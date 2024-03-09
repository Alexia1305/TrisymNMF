function nnlsHALSupdt(U, M, maxiter = 500,V=nothing )
    m, r = size(U)
    UtU = U' * U
    UtM = U' * M
    if V === nothing 
        V = U \ M
        V = max.(V, 0)
        alpha = sum(U' * M .* V) / sum(U' * U .* (V * V'))
        V = alpha * V
    end 
    
    delta = 1e-6  # Condition d'arrêt dépendant de l'évolution de l'itéré V
                  # Arrêter si ||V^{k} - V^{k+1}||_F <= delta * ||V^{0} - V^{1}||_F
                  # où V^{k} est le k-ième itéré.
    eps0 = 0
    cnt = 1
    eps = 1
    while eps >= delta^2 * eps0 && cnt <= maxiter
        nodelta = 0
        if cnt == 1
            eit3 = time()
        end
        for k = 1:r
            
            deltaV = max.((UtM[k, :] - vec(UtU[k, :]' * V)) ./ UtU[k, k], -V[k, :])
            V[k, :] .+= deltaV
            nodelta += sum(deltaV .^ 2)
            if all(V[k, :] .== 0)
                V[k, :] .= 1e-16 * maximum(V)
            end
           
        end
        if cnt == 1
            eps0 = nodelta
        end
        eps = nodelta
        cnt += 1
    end
    return V
end