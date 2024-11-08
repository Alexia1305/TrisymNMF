
Code in julia for the paper :
Dache, Alexandra, Arnaud Vandaele, and Nicolas Gillis. "Orthogonal Symmetric Nonnegative Matrix Tri-Factorization." IEEE International Workshop on Machine Learning for Signal Processing. Institute of Electrical and Electronic Engineers (IEEE), United States, 2024.
# OtrisymNMF

\[
\min_{W \geq 0, S \geq 0} \|X - WSW^T\|_F^2 \quad \text{s.t.} \quad W^TW = I
\]

# OtrisymNMF methods in Julia : algo/OtrisymNMF.jl 

- `OtrisymNMF_CD(X, r, maxiter, epsi, init_algo,time_limit)`

   - `epsi`: criteria : delta error < `epsi`
   - `maxiter`: maximal itterations
   - `init_algo`: "random","k_means","sspa"

# Tests : /Test/ 

- `Test_synt.jl`
- `Test_synt_noise.jl`
- 'CBCL.jl'
- 'TDT2.jl'
