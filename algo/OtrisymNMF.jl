using Random
using LinearAlgebra
using IterTools
using Combinatorics
using Clustering
using Hungarian 

include("septrisymNMF.jl")
include("SSPA.jl")
include("ONMF.jl")



function roots_third_degree(a, b, c, d)
    
    if a == 0
        if b == 0
            root1 = -d/c
            return [root1]
        end
        delta = c^2-4*b*d
        root1 = (-c + sqrt(delta))/ (2 * b)
        root2 = (-c - sqrt(delta))/ (2 * b)
        if root1 == root2
            return [root1]
        else
            return [root1, root2]
        end
    end

    p = -(b^2 / (3 * a^2)) + c / a
    q = ((2 * b^3) / (27 * a^3)) - ((9 * c * b) / (27 * a^2)) + (d / a)
    delta = -(4 * p^3 + 27 * q^2)
    
    if delta < 0
        u = (-q + sqrt(-delta / 27)) / 2
        v = (-q - sqrt(-delta / 27)) / 2
        if u < 0
            u = -(-u)^(1 / 3)
        elseif u > 0
            u = u^(1 / 3)
        else
            u = 0
        end
        if v < 0
            v = -(-v)^(1 / 3)
        elseif v > 0
            v = v^(1 / 3)
        else
            v = 0
        end
        root1 = u + v - (b / (3 * a))
        return [root1]
    elseif delta == 0
        if p == q == 0
            root1 = 0
            return [root1]
        else
            root1 = (3 * q) / p
            root2 = (-3 * q) / (2 * p)
            return [root1, root2]
        end
    else
        epsilon = -1e-300
        phi = acos(-q / 2 * sqrt(-27 / (p^3 + epsilon)))
        z1 = 2 * sqrt(-p / 3) * cos(phi / 3)
        z2 = 2 * sqrt(-p / 3) * cos((phi + 2 * π) / 3)
        z3 = 2 * sqrt(-p / 3) * cos((phi + 4 * π) / 3)
        root1 = z1 - (b / (3 * a))
        root2 = z2 - (b / (3 * a))
        root3 = z3 - (b / (3 * a))
        return [root1, root2, root3]
    end
end
function min_degre4J(a,b,c)
    roots=roots_third_degree(4*a, 0, 2*b, c)
    min=a+b+c
    sol=1
    for x in roots 
        value=a * (x ^ 4) + b * (x ^ 2) + c * x
        if x>0 && value <= min
            sol=x
            min=value
        end 
    end 
    return sol,min
end 

function calcul_erreur(X, W, S)
   
    error2=1-((2*dot(W'*X,S*W')-dot((W'*W)*S,S*(W'*W)))/(dot(X,X)))
    if error2>0 
        error=sqrt(error2)
    else
        error=0
    end 
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

function OtrisymNMF_CD(X, r, maxiter,epsi,init_algo="k_means",time_limit=5)
    debut = time()
    if init_algo=="random"
        # initialisation aléatoire
        n = size(X, 1)
        W = zeros(n, r)
        for i in 1:n
            k = rand(1:r)
            W[i, k] = rand()
        end

        matrice_aleatoire = rand(r, r)
        S = 0.5 * (matrice_aleatoire + transpose(matrice_aleatoire))
        
        
    end 
    if init_algo=="k_means"
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
        
        for k in 1:r
            nw=norm(W[:, k],2)
            if nw==0
                continue
            end     
            W[:, k] .= W[:, k] ./ nw
           
        end
        #OPtimisation de S
        S=W'*X*W
          

    end 
    if init_algo=="spa"
        K = spa(X, r, epsi)
        WO=X[:,K]
        n = size(X, 1)
        norm2x = sqrt.(sum(X.^2, dims=1))
        Xn = X .* (1 ./ (norm2x .+ 1e-16))
        normX2 = sum(X .^ 2)

        e = Float64[]

        HO = orthNNLS(X, WO, Xn)
        W=HO'
        for k in 1:r
            nw=norm(W[:, k],2)
            if nw==0
                continue
            end     
            W[:, k] .= W[:, k] ./ nw
           
        end
        #OPtimisation de S
        S=W'*X*W 
        
        
    end
    if init_algo=="sspa"
        # Initialisation ONMF avec SSPA calcul de X=WOHO et W=HO'
        
        n = size(X, 1)
        p=max(2,Int(floor(0.1*n/r)))
        options = Dict(:average => 1) # Définissez les options avec lra = 1
        WO,K=SSPA(X, r, p, options)

        norm2x = sqrt.(sum(X.^2, dims=1))
        Xn = X .* (1 ./ (norm2x .+ 1e-16))
        normX2 = sum(X .^ 2)

        e = Float64[]

        HO = orthNNLS(X, WO, Xn)
        W=HO'
        for k in 1:r
            nw=norm(W[:, k],2)
            if nw==0
                continue
            end     
            W[:, k] .= W[:, k] ./ nw
           
        end
        #OPtimisation de S
        S=W'*X*W 
        

    end 
    erreur_prec = calcul_erreur(X, W, S)
   
    erreur = erreur_prec
    for itter in 1:maxiter
        # Calculer le temps écoulé
        temps_ecoule = time() - debut

        

        # Vérifier si le temps écoulé est inférieur à la limite
        if temps_ecoule > time_limit
           
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
                x, value_x = min_degre4J(a, b, c)
                if value_x < sum_value && x>0
                    k_min = k
                    k_min_value = x
                    sum_value = value_x
                end
            end
            W[i, :] .= 0
            W[i, k_min] = k_min_value
        end

       
        
        

        for k in 1:r
            nw=norm(W[:, k],2)
            if nw==0
                continue
            end     
            W[:, k] .= W[:, k] ./ nw
           
        end
        #OPtimisation de S
        S=W'*X*W 

        erreur_prec = erreur
        erreur = calcul_erreur(X, W, S)
      
        if erreur<epsi
            break
        end
        if abs(erreur_prec-erreur)<epsi
            break
        end
    end

    
    erreur = calcul_erreur(X, W, S)
    
    return W, S, erreur
end

function OtrisymNMF_MU(X, r, maxiter,epsi,init_alg="k_means",time_limit=5)

    #Orthogonal Nonnegative Matrix Tri-factorizations for Clustering
     # Initialiser le temps de début
    debut = time()
    if init_alg=="random" 
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
    end 
    if init_alg=="k_means" 
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
        R=kmeans(Xnorm,r,maxiter=Int(1000))
       
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
    if init_alg=="spa"
        eps_machine=eps(Float64)
        W, S = septrisymNMF(X, r,epsi)
        n = size(X, 1)
        for i in 1:n
            indice_max = argmax(W[i, :])
            elem=W[i,indice_max]
            W[i, :] .= eps_machine
            W[i, indice_max] = elem
        end 
         # optimisation de S
         WtW=W'*W
         S.=S.*real(sqrt.(Complex.((W'*X*W)./(WtW*S*WtW))))
    end 
    if init_alg=="sspa"
        eps_machine=eps(Float64)
        # Initialisation ONMF avec SSPA calcul de X=WOHO et W=HO'
        
        n = size(X, 1)
        p=max(2,Int(floor(0.2*n/r)))
        options = Dict(:average => 1) # Définissez les options avec lra = 1
        WO,K=SSPA(X, r, p, options)

        norm2x = sqrt.(sum(X.^2, dims=1))
        Xn = X .* (1 ./ (norm2x .+ 1e-16))
        normX2 = sum(X .^ 2)

        e = Float64[]

        HO = orthNNLS(X, WO, Xn)
        W=HO'
        n = size(X, 1)
        for i in 1:n
            indice_max = argmax(W[i, :])
            elem=W[i,indice_max]
            W[i, :] .= eps_machine
            W[i, indice_max] = elem
        end 
        #OPtimisation de S
        S=W'*X*W 
        

    end 
    erreur_prec = calcul_erreur(X, W, S)
    erreur = erreur_prec
    

    for itter in 1:maxiter
        # Calculer le temps écoulé
        temps_ecoule = time() - debut

        

        # Vérifier si le temps écoulé est inférieur à la limite
        if temps_ecoule > time_limit
           
            println("Limite de temps dépassée. mu")
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
        indice_max = argmax(W[i, :])
        elem=W[i,indice_max]
        W[i, :] .= 0
        W[i, indice_max] = elem
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
