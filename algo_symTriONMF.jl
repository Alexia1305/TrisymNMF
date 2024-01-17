using Random
using LinearAlgebra
using IterTools
using Combinatorics
using Clustering

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
   
    error=1-((2*dot(W'*X,S*W')-dot((W'*W)*S,S*(W'*W)))/(dot(X,X)))
    return error
end

function calcul_accuracy(W_true,W_find)
    dim = size(W_true)
    n=dim[1]
    r=dim[2]
    vecteur_original=1:r
    toutes_permutations = collect(permutations(vecteur_original))

    maxi=-Inf
    for perm in toutes_permutations
        accuracy=float(0)
        for k in 1:r
            accuracy += count(x -> x[1] != 0 && x[2] != 0 , zip(W_true[:, k], W_find[:, perm[k]]))
        end
        if accuracy>maxi
            maxi=accuracy
        end 
    end 
    maxi /= float(n)
    return float(maxi)

end 

function symTriONMF_coordinate_descent(X, r, maxiter,epsi,init_kmeans)
    debut = time()
    if init_kmeans == false 
        # initialisation aléatoire
        n = size(X, 1)
        W = zeros(n, r)
        for i in 1:n
            k = rand(1:r)
            W[i, k] = rand()
        end

        matrice_aleatoire = rand(r, r)
        S = 0.5 * (matrice_aleatoire + transpose(matrice_aleatoire))
    else 
        # initialisation kmeans 
        R=kmeans(X',r,maxiter=Int(ceil(maxiter*0.03)))
        a = assignments(R)  
        n = size(X, 1)
        W = zeros(n, r)
        for i in 1:n
            W[i, a[i]] = 1
        end
        matrice_aleatoire = rand(r, r)
        S = 0.5 * (matrice_aleatoire + transpose(matrice_aleatoire))
        #optimisation de S
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

    end 
    erreur_prec = calcul_erreur(X, W, S)
    erreur = erreur_prec
    

    for itter in 1:maxiter
        # Calculer le temps écoulé
        temps_ecoule = time() - debut

        

        # Vérifier si le temps écoulé est inférieur à la limite
        if temps_ecoule > 1
           
            println("Limite de temps dépassée.")
            break
        end
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
   
   
    return W, S, erreur
end

function symTriONMF_update_rules(X, r, maxiter,epsi,init_kmeans)

    #Orthogonal Nonnegative Matrix Tri-factorizations for Clustering
     # Initialiser le temps de début
    debut = time()
    if init_kmeans == false 
        # initialisation aléatoire
        n = size(X, 1)
        eps_machine=1e-5
        W = fill(eps_machine, (n,r))
        for i in 1:n
            k = rand(1:r)
            W[i, k] = rand()
        end

        matrice_aleatoire = rand(r, r)
        S = 0.5 * (matrice_aleatoire + transpose(matrice_aleatoire))
    else 
        # initialisation kmeans 
        R=kmeans(X',r,maxiter=Int(ceil(maxiter*0.03)))
        a = assignments(R)  
        n = size(X, 1)
        eps_machine=eps(Float64)
        W = fill(eps_machine, (n,r))
        for i in 1:n
            W[i, a[i]] = 1
        end
        matrice_aleatoire = rand(r, r)
        S = 0.5 * (matrice_aleatoire + transpose(matrice_aleatoire))
        #optimisation de S
        # optimisation de S
        WtW=W'*W
        S.=S.*real(sqrt.(Complex.((W'*X*W)./(WtW*S*WtW))))
         
        

    end 
    erreur_prec = calcul_erreur(X, W, S)
    erreur = erreur_prec
    

    for itter in 1:maxiter
        # Calculer le temps écoulé
        temps_ecoule = time() - debut

        

        # Vérifier si le temps écoulé est inférieur à la limite
        if temps_ecoule > 1
           
            println("Limite de temps dépassée.")
            break
        end
        
        # optimisation de W
        # Calcul de XtWS
        XtWS = X' * (W * S)

        # Calcul de la mise à jour de W
        W .= W .* real(sqrt.(Complex.(XtWS ./ (W * (W' * XtWS)))))

       
        

        # optimisation de S
        WtW=W'*W
        S.=S.*real(sqrt.(Complex.((W'*X*W)./(WtW*S*WtW))))
       

        erreur_prec = erreur
        erreur = calcul_erreur(X, W, S)
      
        if erreur<epsi
            break
        end
        if abs(erreur_prec-erreur)<epsi
            break
        end
    end
    for i in 1:n
        for j in 1:r 
            if W[i,j]<=eps_machine
                W[i,j]=0
            end
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
   
    erreur2 = calcul_erreur(X, W, S)
      
    return W, S, erreur2
end

