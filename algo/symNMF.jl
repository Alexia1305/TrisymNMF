using Distributions
using LinearAlgebra
using SparseArrays
using Hungarian
function frobenius_sym_loss(A::Matrix{Float64}, M, MA::Matrix{Float64})
    t1 = 0.5 * (norm(M) ^ 2 + norm(A' * A) ^ 2) # A' * A is a small r x r sized matrix
    
    mul!(MA, M, A)
    return t1 - dot(A, MA)
 end
function pgradnorm_SNMF(grad::Matrix{Float64}, A::Matrix{Float64})
    t1 = reduce(+, grad[i] ^ 2 for i in 1:length(A) if A[i] > 0.; init = 0.)
    t2 = reduce(+, min(0., grad[i]) ^ 2 for i in 1:length(A) if A[i] == 0.; init = 0.)
    t1 + t2
end
function random_init_sym(M, r::Int)
    n = size(M)[1]
    scale = sqrt(mean(M) / float(r))

    A = 2. * rand(Uniform(1e-5, scale), n, r)
    print(size(A))
    return A
end
function random_init_kmeans(X, r::Int)
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
    W = zeros(n, r)
    for i in 1:n
        W[i, a[i]] = 1
    end
    println(size(W))
    
    return W
end
function grad_SNMF!(G::Matrix{Float64}, M, A::Matrix{Float64}, MA::Matrix{Float64})
    mul!(MA, M, A)
    mul!(G, A, A' * A)
    @. G = 2 * (G - MA)
end

"""Computes the gradient of the SNMF objective function
    0.5 * |M - A * A'|^2
"""
function grad_SNMF(M, A::Matrix{Float64})
    MA = M * A
    AAtA = A * (A' * A) # order is important for efficient computation
    return 2 * (AAtA - MA)
end

function poly4(a::Float64, b::Float64, x::Float64)
    return 0.25 * x^4 + a * 0.5 *  x ^ 2 + b * x
end

function cubic_root(x)
    x = real(x)
   if x >= 0.
        return x ^ (1. / 3.)
    else
        return - (abs(x) ^ (1. / 3.))
    end
end

"""Returns argmin_{x >= 0} x^4 / 4 + a x ^ 2 / 2 + b x
"""
function best_poly_root(a::Float64, b::Float64)
    delta = 4 * a ^ 3 + 27 * b ^ 2
    d = 0.5 * (-b + sqrt(Complex(delta / 27.)))

    if delta <= 0.
        r = 2 * cubic_root(abs(d))
        theta = angle(d) / 3.
        best_z = 0.
        best_y = 0.

        for k = 0:2
            z = r * cos(theta + 2 * k * pi / 3.)
            if (z >= 0.) && (poly4(a, b, z) < best_y)
                best_z = z
                best_y = poly4(a, b, z)
            end
        end

        return best_z
    else
        z_c = cubic_root(d) + cubic_root(0.5 * (- b - sqrt(delta / 27.)))
        z = Real(z_c)
        if (z >= 0.) && (poly4(a, b, z) < 0.)
            return z
        else
            return 0.
        end
    end
end

   
   
"""Does a full iteration of coordinate descent SymNMF.
Reference:
    A. Vandaele, N.Gillis et al.
    Efficient and non-convex coordinate descent for symmetric nonnegative matrix factorization.
    IEEE Transactions on Signal Processing, 2016
"""
function update_CD!(M, A::Matrix{Float64}, MA::Matrix{Float64}, grad::Matrix{Float64},
    A_coeffs::Matrix{Float64}, B_coeffs::Matrix{Float64},
    C::Vector{Float64}, L::Vector{Float64}, D::Matrix{Float64})

    n, r = size(A)
    
    for j = 1:r
        for i = 1:n
            A_coeffs[i,j] = C[j] + L[i] - 2. * A[i,j] ^ 2 - M[i,i]
          
            Ai = @view(A[i,:])
            Dj = @view(D[:,j])
            Aj = @view(A[:,j])
            Mi = @view(M[:,i])
            B_coeffs[i,j] = dot(Ai, Dj) - dot(Aj, Mi)
            B_coeffs[i,j] = B_coeffs[i,j] - A[i,j] ^ 3 - A[i,j] * A_coeffs[i,j]
            Aij_new = best_poly_root(A_coeffs[i,j], B_coeffs[i,j])

            C[j] = C[j] + Aij_new ^ 2 - A[i,j] ^ 2
            L[i] = L[i] + Aij_new ^ 2 - A[i,j] ^ 2

            @. D[j,:] = D[j,:] + A[i,:] * (Aij_new - A[i,j])
            @. D[:,j] = D[j,:]
            D[j,j] = C[j]
            A[i,j] = Aij_new

        end
    end

    grad_SNMF!(grad, M, A, MA)
    pg_norm = pgradnorm_SNMF(grad, A)
    return pg_norm
end

"""Generic wrapper for all SymNMF solvers.
Arguments

    - M::GenMatrix : the symmetric nonnegative matrix to factorize
    - r::Int: the target rank of the factorization

Keyword optionnal arguments
    - algo::Symbol : the algorithm to use.

        :pga : Projected Gradient method with Armijo line search
        :nolips : NoLips with fixed step size rule
        :dyn_nolips : NoLips with dynamical step size strategy
        :adap_nolips : NoLips with adaptive coefficients
        :beta : Beta-SNMF
        :cd : coordinate descent
        :sym_hals
        :sym_anls

    - max_iter::Int : maximal number of iterations
    - max_iter::Float64 : maximal running time (seconds)
    - monitoring_interval::Float64 : the interval between evaluations of objective
        and/or clustering accuracy. Set to 0 for no monitoring

    - A_init::GenMatrix : initial value of matrix A
    - monitor_accuracy : set to true to measure the clustering accuracy with respect
        to the true labels
    - true_labels::Vector{Int64} : ground truth labels of the dataset

Optionnal algorithm-specific parameters can also be passed via keyword arguments.s

Output
    - A::Matrix{Float64} : the solution matrix A
    - losses::Matrix{Float64} : a (n_measures, 4) size matrix where
        losses[:,1] is the time of the measures
        losses[:,2] are the values of the objective function
        losses[:,3] the convergence measure
        losses[:,4] the clustering accuracies
"""
function SymNMF(M, r::Int;max_iter=500, max_time=5,tol = 1e-7, A_init = "random")
    monitoring_interval=0
    n = size(M, 1)

    # initialization
    if A_init =="k_means"
        A = random_init_kmeans(M, r)
    else
        A = random_init_sym(M, r)
    end

    # for faster computations we store also the matrix M'
    if typeof(M) == SparseMatrixCSC{Float64, Int}
        Mt = sparse(M')
    else
        Mt = Matrix(M')
    end
   
    MA = zeros(size(A))
    grad = zeros(size(A))
   
    grad_SNMF!(grad, M, A, MA)
    initial_pgnorm = max(pgradnorm_SNMF(grad, A), 1e-16)
    pg_norm = initial_pgnorm

    losses = Array{Float64}(undef, 0, 4)
    t0 = time_ns()
    t_prev = t0

    keep_going = true
    pgnorm_cond = true
    it = 0

    

    
    C = sum(abs2, A, dims = 1)[:]
    L = sum(abs2, A, dims = 2)[:]
    D = A' * A
    A_coeffs = zeros(size(A))
    B_coeffs = zeros(size(A))

    

    while keep_going
        # monitoring loss
        if (monitoring_interval > 0.) && (float(time_ns() - t_prev) / 1e9 >= monitoring_interval) || (it == 0)
            delta_t = float(time_ns() - t_prev) / 1e9
            loss = frobenius_sym_loss(A, M, MA)
            clust_acc = 0.

            

            losses = vcat(losses, [delta_t loss (pg_norm / initial_pgnorm) clust_acc])
            t_prev = time_ns()
        end

       
    
        pg_norm = update_CD!(M, A, MA, grad, A_coeffs, B_coeffs, C, L, D)
        

        # checking stopping criterion
        it += 1
        time_cond = (time_ns() - t0) / 1e9 < max_time
        keep_going = (it <= max_iter) && time_cond && pg_norm / initial_pgnorm > tol

    end

#     if (algo == :sym_hals) || (algo == :sym_anls) || (algo == :admm)
#        println("Constraint satisfaction for penalty method $algo : |A - B|/|A| = ",
#         norm(A - Bt) / norm(A))
#     end
                                                        
#     println("Terminated after $it iterations.")
    error=norm(M-A*A',2)/norm(M,2)
    return A, error 
end
