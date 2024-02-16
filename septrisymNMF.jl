using SparseArrays
using Random
using LinearAlgebra
using MAT
function nnls(A, b)
    m, n = size(A)
    λ = 0.1  # Regularization parameter
    c = ones(n)
    x = zeros(n)
    iter = 0
    max_iter = 1000
    tol = 1e-6
    while iter < max_iter
        iter += 1
        x_old = copy(x)
        r = b - A * x
        indices = findall(x -> x > 0, c)
        Asub = A[:, indices]
        x[indices] .= max.(0, x[indices] .+ (Asub \ r))
        if norm(x - x_old) < tol
            break
        end
    end
    return x
end

function spa(X::AbstractMatrix{T},
    r::Integer,
    epsilon::Float64=10e-9) where T <: AbstractFloat
# Get dimensions
m, n = size(X)
col_sums = sum(X, dims=1)

# Calculate the inverse of the column sums and transpose to get a row vector
inverse_col_sums = 1. ./ col_sums'
D=diagm(vec(inverse_col_sums))
X=X*D
# Set of selected indices
K = zeros(Int, r)

# Norm of columns of input X
normX0 = sum.(abs2,eachcol(X))
# Max of the columns norm
nXmax = maximum(normX0)
# Init residual
normR = copy(normX0)
# Init set of extracted columns
U = Matrix{T}(undef,m,r)

i = 1

while i <= r && sqrt(maximum(normR)/nXmax) > epsilon    
# Select column of X with largest l2-norm
a = maximum(normR)
# Check ties up to 1e-6 precision
b = findall((a .- normR) / a .<= 1e-6)
# In case of a tie, select column with largest norm of the input matrix
_, d = findmax(normX0[b])
b = b[d]
# Save index of selected column, and column itself
K[i] = b
U[:,i] .= X[:,b]
# Update residual
for j in 1:i-1
   U[:,i] .= U[:,i] - U[:,j] * (U[:,j]' * U[:,i])
end
U[:,i] ./= norm(U[:,i])
normR .-= (X'*U[:,i]).^2
# Increment iterator
i += 1
end

return K
end

function calcul_erreur(X, W, S)
   
    error=1-((2*dot(W'*X,S*W')-dot((W'*W)*S,S*(W'*W)))/(dot(X,X)))
    return error
end
# Define function for separable tri-symNMF
function septrisymNMF(A, r, epsilon)
    n = size(A, 1)
    
    # Idendity K such that A(K,K) = W(K,:) S W(K,:)^T
    K = spa(A, r, epsilon)
    W = zeros(n,r)
    # Solve A(K,K) z = q = A(K,:)e
    q = sum(A[:, K], dims=1)'
    y = nnls(A[K, K], q)
    
    # Recover S and W
    S = diagm(y) * A[K, K] * diagm(y)
    for i in 1:n

        W[i,:]= nnls((diagm(vec(y)) * A[K, K])', A[K, i])'
    end 
    return W, S
end

# Example usage
epsilon=10e-12 # Define options dictionary
# Generate a synthetic W and S, where W is separable
r = 5


# Charger le fichier MAT
matfile = matread("A.mat")

# Accéder à la matrice A dans le fichier
A = matfile["A"]

Wt, St = septrisymNMF(A, r, epsilon)
println(norm(A-Wt*St*Wt',2))
