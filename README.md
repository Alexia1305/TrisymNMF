# SymTriNMF

![Equation](https://render.githubusercontent.com/render/math?math=%5Cmin_%7BW%20%5Cgeq%200%2C%20S%20%5Cgeq%200%7D%20%5C%7CX%20-%20WSW%5ET%5C%7C_F%5E2%20%5Cquad%20%5Ctext%7Bs.t.%7D%20%5Cquad%20W%5ETW%20%3D%20I)

# Algorithme de descente en coordonnées en Julia : algo_symTriONMF.jl 

- `symTriONMF_coordinate_descent(X, r, maxiter, epsi, init_kmeans)`

   - `epsi`: critère de convergence, delta error < `epsi`
   - `maxiter`: nombre maximal d'itérations
   - `init_kmeans`: true pour initialisation par kmeans, false pour une initialisation aléatoire

# Tests synthétiques

- `Test_synt.jl` et `Test_synt_noise.jl`
