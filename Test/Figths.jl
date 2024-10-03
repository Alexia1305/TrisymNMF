include("../algo/OtrisymNMF.jl")
using DelimitedFiles
using Plots
gr()  # Configurer le backend GR
using SparseArrays
using LaTeXStrings
using Random
Random.seed!(213)


using Statistics

n = 100
r = 5
nbr_test = 100
errors = zeros(nbr_test, 4)
times = zeros(nbr_test, 4)

for test in 1:nbr_test
    println("Test: ", test)
    
    
    X=zeros(n,n)
    team=zeros(1,n)
    esp=Int(floor(n/r))
    for i in 1:r
        # Générer des indices aléatoires pour les lignes 1 à 40 et les colonnes 41 à 200
        rows_indices = rand((i-1)*esp+1:i*esp,300)  # Génère 100 indices aléatoires entre 1 et 40
        cols_indices = rand(setdiff(1:n, (i-1)*esp+1:i*esp), 300)  # Génère 100 indices aléatoires entre 41 et 200

        # Mettre des 1 aux positions spécifiées par les indices
        X[rows_indices, cols_indices] .= 1
        X[cols_indices, rows_indices] .= 1

    end 
    println(X)
    indices_melanges = shuffle(1:n)
    X=X[indices_melanges,indices_melanges]
    team=team[1,indices_melanges]

    maxiter=10000
    epsi=10e-7
    init="sspa"
    timelimit=50
    
    times[test,1] = @elapsed begin
        Wb, S, erreur = OtrisymNMF_CD(X, r, maxiter, epsi,"k_means", timelimit)
    end
    println(erreur)
    println(S)
   errors[test,1] = erreur
   
    times[test,2]  = @elapsed begin
        W, H, erreur = alternatingONMF(X, r, maxiter, epsi,init)
    end
   errors[test,2] = erreur

    times[test,1]  = @elapsed begin
        A, erreur = SymNMF(X, r; max_iter=maxiter, max_time=timelimit, tol=epsi, A_init=init)
    end
    errors[test,3] = erreur
    times[test,4]= @elapsed begin
        W,S, erreur = OtrisymNMF_MU(X, r, maxiter, epsi, init,timelimit)
    end
    errors[test,4] = erreur
end

# Calculate means and standard deviations
mean_errors = mean(errors, dims=1)
std_errors = std(errors, dims=1)
mean_times = mean(times, dims=1)
std_times = std(times, dims=1)

# Print and write results for each method
nom_fichier = "resultats_figths.txt"

open(nom_fichier, "w") do f
    write(f, "CD:\n")
    write(f, "Mean Error: $(mean_errors[1]), Standard Deviation of Error: $(std_errors[1]), Mean Time: $(mean_times[1]), Standard Deviation of Time: $(std_times[1])\n\n")
    
    write(f, "ONMF:\n")
    write(f, "Mean Error: $(mean_errors[2]), Standard Deviation of Error: $(std_errors[2]), Mean Time: $(mean_times[2]), Standard Deviation of Time: $(std_times[2])\n\n")
    
    write(f, "Symnmf:\n")
    write(f, "Mean Error: $(mean_errors[3]), Standard Deviation of Error: $(std_errors[3]), Mean Time: $(mean_times[3]), Standard Deviation of Time: $(std_times[3])\n\n")
    
    write(f, "Mu:\n")
    write(f, "Mean Error: $(mean_errors[4]), Standard Deviation of Error: $(std_errors[4]), Mean Time: $(mean_times[4]), Standard Deviation of Time: $(std_times[4])\n\n")
end

# Print results
println("CD:")
println("Mean Error: ", mean_errors[1])
println("Standard Deviation of Error: ", std_errors[1])
println("Mean Time: ", mean_times[1])
println("Standard Deviation of Time: ", std_times[1])
println()

println("ONMF:")
println("Mean Error: ", mean_errors[2])
println("Standard Deviation of Error: ", std_errors[2])
println("Mean Time: ", mean_times[2])
println("Standard Deviation of Time: ", std_times[2])
println()

println("symnmf:")
println("Mean Error: ", mean_errors[3])
println("Standard Deviation of Error: ", std_errors[3])
println("Mean Time: ", mean_times[3])
println("Standard Deviation of Time: ", std_times[3])
println()

println("MU:")
println("Mean Error: ", mean_errors[4])
println("Standard Deviation of Error: ", std_errors[4])
println("Mean Time: ", mean_times[4])
println("Standard Deviation of Time: ", std_times[4])