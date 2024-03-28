using Distributions
using LinearAlgebra
using SparseArrays


function svds(A, k)
    U, Σ, V = svd(A)
    Uk = U[:, 1:k]
    Σk = Diagonal(Σ[1:k])
    Vk = V[:, 1:k]
    return Uk, Σk, Vk
end

function updateorthbasis(V, v)
    if isempty(V)
        V = v / norm(v,2)
    else
        # Project new vector onto orthogonal complement, and normalize
        v -= V * (V' * v)
        v /= norm(v)
        V = hcat(V, v)
    end
    return V
end
function colaverage(W, type)
    if size(W, 2) == 1
        return W
    else
        if type == 1
            return mean(W, dims=2)
        elseif type == 2
            return median(W, dims=2)
        elseif type == 3
            u, s, v = svds(W, nsv=1)
            if sum(v[v .> 0]) < sum(v[v .< 0])
                u = -u
                v = -v
            end
            v = s .* v
            return u * mean(v)
        elseif type == 4
            u, v = L1LRAcd(W, 1, 100)
            return u * median(v)
        end
    end
end


function SSPA(X, r, p, options)
    if !haskey(options, :lra)
        options[:lra] = 0
    end
    
    if options[:lra] == 1
        Y, S, Z = svds(X, nsv=r)
        Z = S * Z'
    else
        Z = X
    end

    if !haskey(options, :average)
        options[:average] = 0
    end
    
    V = []
    normX2 = sum(X.^2, dims=1)
    W = zeros(size(Z, 1), r)
    K = zeros(Int, r, p)
    
    for k in 1:r
        spa = argmax(normX2)
        diru = X[:, spa.I[2]]
        
        if k >= 2
            diru -= V * (V' * diru)
        end
        
        u = diru' * X
        b = sortperm(vec(u), rev=true)
        K[k, :] = b[1:p]
        
        if p == 1
            W[:, k] = Z[:, K[k, :]]
        else
            if options[:average] == 1
                W[:, k] = mean(Z[:, K[k, :]], dims=2)
            elseif options[:average] == 3
                W[:, k] = colaverage(Z[:, K[k, :]], 3)
            elseif options[:average] == 4
                W[:, k] = colaverage(Z[:, K[k, :]], 4)
            else
                W[:, k] = median(Z[:, K[k, :]], dims=2)
            end
        end
        
        V = updateorthbasis(V, W[:, k])
        normX2 .-= (transpose(V[:, end]) * X).^2
    end
    
    if options[:lra] == 1
        W = Y * W
    end
    
    return W, K
end