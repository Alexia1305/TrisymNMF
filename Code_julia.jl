using Random
using LinearAlgebra

function min_degre4(a, b, c)
    # minimum du polynome de degré 4 ax^4 + bx^2 + cx
    # car pas 0 
    min = 1
    sol = a+b+c
    if a == 0
        x = -c / (2 * b)
        value = a * (x ^ 4) + b * (x ^ 2) + c * x
        if x > 0 && value < min
            sol = x
            min = value
        end
        return sol, min
    end

    # méthode de cardan
    p = 2 * b / (4 * a)
    q = c / (4 * a)
    delta = -(4 * (p ^ 3) + 27 * (q ^ 2))

    if delta < 0
        x = cbrt((-q + sqrt(-delta / 27)) / 2) + cbrt((-q - sqrt(-delta / 27)) / 2)
        value = a * (x ^ 4) + b * (x ^ 2) + c * x
        if x > 0 && value < min
            sol = x
            min = value
        end
    elseif delta == 0
        if p == q && q == 0
            x = 0
            value = 0
        else
            x = 3 * q / p
            value = a * (x ^ 4) + b * (x ^ 2) + c * x
            if x > 0 && value < min
                sol = x
                min = value
            end
            x = -3 * q / (2 * p)
            value = a * (x ^ 4) + b * (x ^ 2) + c * x
            if x > 0 && value < min
                sol = x
                min = value
            end
        end
    else
        for k in 0:2
            x = 2 * sqrt(-p / 3) * cos(
                (1 / 3) * acos((3 * q / (2 * p)) * ((3 / -p) ^ (1 / 2))) + (2 * k * π / 3)
            )
            value = a * (x ^ 4) + b * (x ^ 2) + c * x
            if x > 0 && value < min
                sol = x
                min = value
            end
        end
    end
    return sol, min
end

function calcul_erreur(X, W, S)
    # Calcul de X - WSW^T
    result_matrix = X - W * S * transpose(W)

    # Calcul de la norme de Frobenius au carré
    erreur = sum(result_matrix .^ 2) / sum(X .^ 2)
    return erreur
end

function symTriONMF_coordinate_descent(X, r, maxiter,epsi)
    # initialisation aléatoire
    n = size(X, 1)
    W = zeros(n, r)
    for i in 1:n
        k = rand(1:r)
        W[i, k] = rand()
    end

    matrice_aleatoire = rand(r, r)
    S = 0.5 * (matrice_aleatoire + transpose(matrice_aleatoire))

    erreur_prec = calcul_erreur(X, W, S)
    erreur = erreur_prec
    

    for itter in 1:maxiter
        # optimisation de W
        for i in 1:n
            k_min = nothing
            k_min_value = nothing
            sum_value = Inf
            for k in 1:r
                a = S[k, k] ^ 2
                b = -2 * X[i, i] * S[k, k]
                c = 0
                for j in 1:n
                    if j == i
                        continue
                    end
                    l = argmax(W[j, :])  # l'élément non nul
                    b += 2 * (S[k, l] * W[j, l]) ^ 2
                    c += -4 * X[i, j] * S[k, l] * W[j, l]
                end
                x, value_x = min_degre4(a, b, c)
                if value_x < sum_value && x>0
                    k_min = k
                    k_min_value = x
                    sum_value = value_x
                end
            end
            W[i, :] .= 0
            W[i, k_min] = k_min_value
        end

        erreur_prec = erreur
        erreur = calcul_erreur(X, W, S)
        println("W ", erreur)

        # optimisation de S
        for k in 1:r
            for l in 1:r  # symétrique
                a = 0
                b = 0
                ind_i = findall(W[:, k] .> 0)
                if isempty(ind_i)
                    break
                end
                ind_j = findall(W[:, l] .> 0)
                if isempty(ind_j)
                    break
                end
                for i in ind_i
                    for j in ind_j
                        a += (W[i, k] * W[j, l]) ^ 2
                        b += 2 * X[i, j] * W[i, k] * W[j, l]
                    end
                end
                S[k, l] = max(b / (2 * a), 0)
            end
        end

        erreur_prec = erreur
        erreur = calcul_erreur(X, W, S)
        println("error_S ", erreur)
        if erreur<epsi
            break
        end
        if abs(erreur_prec-erreur)<epsi
            break
        end
    end

    for k in 1:r
        nw=norm(W[:, k],2)
        if nw==0
            continue
        end     
        W[:, k] .= W[:, k] ./ nw
        
        S[k, :] .= S[k, :] .* nw
        S[:, k] .= S[:, k] .* nw
        
       
    end
    println("nvll",calcul_erreur(X, W, S))
    return W, S, erreur
end

# création matrice
n = 20
r = 5
W = zeros(n, r)
for i in 1:n
    k = rand(1:r)
    W[i, k] = rand()
end
matrice_aleatoire = rand(r, r)
S = 0.5 * (matrice_aleatoire + transpose(matrice_aleatoire))

X = W * S * transpose(W)
println(X)
maxiter=1000
epsi=10e-5
# algorithme :
W, S, erreur = symTriONMF_coordinate_descent(X, r, maxiter,epsi)
println(X)
println(W)
println(S)
println(W * S * transpose(W))
println(W'*W)