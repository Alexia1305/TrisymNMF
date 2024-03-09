using SparseArrays
using Random
using LinearAlgebra
using MAT
include("NNLS.jl")

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
# Define function for separable tri-symNMF
function septrisymNMF(A, r, epsilon=10e-12)
    n = size(A, 1)
    
    # Idendity K such that A(K,K) = W(K,:) S W(K,:)^T
    K = spa(A, r, epsilon)
    W = zeros(n,r)
    # Solve A(K,K) z = q = A(K,:)e
    q = sum(A[:, K], dims=1)[:]'
    y = nnlsHALSupdt(A[K, K], reshape(vec(q), length(vec(q)), 1))
    # Recover S and W
    S = diagm(vec(y)) * A[K, K] * diagm(vec(y))
    

    W= nnlsHALSupdt(Matrix((diagm(vec(y)) * A[K, K])'), A[K, :])'
     
    return W, S
end

