# OtrisymNMF

\[
\min_{W \geq 0, S \geq 0} \|X - WSW^T\|_F^2 \quad \text{s.t.} \quad W^TW = I
\]

# Algorithme de descente de coordonnées en Julia : algo_symTriONMF.jl 

- `OtrisymNMF_coordinate_descent(X, r, maxiter, epsi, init_kmeans)`

   - `epsi`: critère de convergence, delta error < `epsi`
   - `maxiter`: nombre maximal d'itérations
   - `init_kmeans`: true pour initialisation par kmeans, false pour une initialisation aléatoire

# Tests synthétiques

- `Test_synt.jl` et `Test_synt_noise.jl`
