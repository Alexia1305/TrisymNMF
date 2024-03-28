include("SNPA")
include("SSPA.jl")
using LinearAlgebra


function orthNNLS(M, U, Mn)
    if Mn === nothing
        norm2m = sqrt.(sum(M.^2, dims=1))
        Mn = M .* (1 ./ (norm2m .+ 1e-16))
    end
    m, n = size(Mn)
    m, r = size(U)
    norm2u = sqrt.(sum(U.^2, dims=1))
    Un = U .* (1 ./ (norm2u .+ 1e-16))
    A = Mn' * Un
    b = argmax(A', dims=1)
    V = zeros(r, n)
    for i in 1:n
        idx = b[1, i].I[1]
        V[idx, i] = sum(M[:, i] .* U[:, idx]) ./ norm(U[:, idx],2)^2
    end
    return V
end



function alternatingONMF(X, r, maxiter=100,delta=1e-6,init_algo="k_means")
    if init_algo=="k_means"
        # initialisation kmeans 
        n = size(X, 1)
        Xnorm = similar(X, Float64)
        # Pour chaque colonne de X
        for i in 1:n
            # Calcul de la norme euclidienne de la colonne
            col_norm = norm(X[:, i],2)
            # Division de la colonne par sa norme
            if col_norm !=0
                Xnorm[:, i] = X[:, i] / col_norm
            else
                Xnorm[:, i] = X[:, i] 
            end  
        end
        R=kmeans(Xnorm,r,maxiter=Int(ceil(1000)))
        a = assignments(R)  
        n = size(X, 1)
        H = zeros(r, n)
        for i in 1:n
            H[a[i], i] = 1
        end
        #orthogonalisation des lignes 
        for k in 1:r
            nw=norm(H[k, :],2)
            if nw==0
                continue
            end     
            H[k, :] .= H[k, :] ./ nw
           
        end
        W = X * H'
    end 
    #initialisation snpa :
    if init_algo =="snpa"

        K = snpa(X, r ; normalize=true)
        W = X[:, K]
        if length(K) < r
            error("SNPA was not able to extract r indices. This means that your data set does not even have r extreme rays.")
        end
    end 
    if init_algo=="sspa"
        n = size(X, 1)
        p=max(2,Int(floor(0.1*n/r)))
        options = Dict(:average => 1) # Définissez les options avec lra = 1
        W,K=SSPA(X, r, p, options)
    end 

    m, n = size(X)
    m, r = size(W)
    norm2x = sqrt.(sum(X.^2, dims=1))
    Xn = X .* (1 ./ (norm2x .+ 1e-16))
    normX2 = sum(X .^ 2)
    k = 1
    
    e = Float64[]
    H=zeros(r,n)
    while k <= maxiter && (k <= 3 || abs(e[end - 1] - e[end - 2]) > delta)
        H = orthNNLS(X, W, Xn)
        norm2h = sqrt.(sum(H'.^2, dims=1)) .+ 1e-16
        H = (1 ./ norm2h') .* H
        W = X * H'
        err = (normX2 - sum(W .^ 2)) / normX2
        if err<0
            err=0
        else
            err=sqrt(err)
        end 
        push!(e, err)
        
        k += 1
    end
    
    return W, H, e[end]
end

