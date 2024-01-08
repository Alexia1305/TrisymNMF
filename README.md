# SymTriNMF
\[ \min_{W \geq 0, S \geq 0} \|X - WSW^T\|_F^2 \quad \text{s.t.} \quad W^TW = I \]
 

# algo coordiante descent en julia : algo_symTriONMF.jl 
- symTriONMF_coordinate_descent(X, r, maxiter,epsi,init_kmeans)
epsi : delta error <epsi
maxiter : max of itteration 
init_kmeans : true initialisation by kmeans , flase: random

# test synthétique : Test_synt.jl and Test_synt_noise.jl
