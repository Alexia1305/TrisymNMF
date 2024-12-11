using Random
using LinearAlgebra
using IterTools
using Combinatorics
using Clustering
using Hungarian 

include("septrisymNMF.jl")
include("SSPA.jl")
include("ONMF.jl")

function cardan(a, b, c, d)
    
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


function calcul_erreur(X, S, w, v)
    n = size(X, 1)
    f = 0.0

    # Récupérer les indices et les valeurs non nulles de la matrice sparse X
    rows, cols, vals = findnz(X)

    # Parcourir uniquement les éléments non nuls de X
    for k in 1:length(vals)
        f += (vals[k] - S[v[rows[k]], v[cols[k]]] * w[rows[k]] * w[cols[k]])^2
    end

    # Normaliser l'erreur par la norme de Frobenius de X
    f = sqrt(f) / norm(X)

    return f
end
    

function UpdateW(X, S, w, v)
    n = size(X, 1)
    r = size(S, 1)
    
    # Pré-calculs pour éviter une double boucle sur "n"
    wp2 = zeros(r)
    for k in 1:r
        for p in 1:n
            wp2[k] += (w[p] * S[v[p], k])^2
        end
    end

    # Mise à jour de W
    for i in 1:n
        vi_new = -1
        wi_new = -1.0
        f_new = Inf

        for k in 1:r
            # Minimisation : résoudre 4*c3*x^3 + 2*c1*x + c0 = 0
            c3 = S[k, k]^2
            c2 = 0
            c1 = 2 * (wp2[k] - (w[i] * S[v[i], k])^2) - 2 * S[k, k] * X[i, i]
            c0 = 0

            # Obtenir les indices des colonnes non nulles dans la ligne i de X
            cols = findall(x -> x != 0, X[i, :])

            # Parcourir uniquement les colonnes non nulles
            for p in cols
                if p != i
                    c0 += X[i, p] * w[p] * S[v[p], k]
                end
            end
            c0 = -4 * c0

            # Résolution de l'équation polynomiale avec les racines de Cardan
            roots = cardan(4 * c3, c2, 2 * c1, c0)

            # Identifier la meilleure solution
            x = 1.0
            min_value = c3 * x^4 + c1 * x^2 + c0 * x

            for sol in roots
                value = c3 * sol^4 + c1 * sol^2 + c0 * sol
                if sol > 0 && value < min_value
                    x = sol
                    min_value = value
                end
            end

            if c3 * x^4 + c1 * x^2 + c0 * x < f_new
                f_new = c3 * x^4 + c1 * x^2 + c0 * x
                wi_new = x
                vi_new = k
            end
        end

        # Mise à jour de wp2 avant d'actualiser w[i]
        for k in 1:r
            wp2[k] = wp2[k] - (w[i] * S[v[i], k])^2 + (wi_new * S[vi_new, k])^2
        end

        # Mise à jour de w[i] et v[i]
        w[i] = wi_new
        v[i] = vi_new
    end

    # Normalisation des colonnes de w
    nw = zeros(r)
    for i in 1:n
        nw[v[i]] += w[i]^2
    end
    for k in 1:r
        nw[k] = sqrt(nw[k])
    end
    for i in 1:n
        w[i] /= nw[v[i]]
    end

    return w, v
end

function UpdateS(X, r, w, v)
    # Initialiser S comme une matrice de zéros de taille r x r
    S = zeros(r, r)
    
    # Récupérer les indices et les valeurs non nulles de la matrice sparse X
    rows, cols, vals = findnz(X)

    # Parcourir uniquement les éléments non nuls de X
    for k in 1:length(vals)
        S[v[rows[k]], v[cols[k]]] += w[rows[k]] * w[cols[k]] * vals[k]
    end

    return S
end

function OtrisymNMF_CD(X, r, maxiter,epsi,init_algo="k_means",time_limit=5)
    debut = time()
    n=size(X,1)
    w=zeros(n,1)
    v=zeros(Int,n)
    if init_algo=="random"
        
        for i in 1:n
            v[i] = rand(1:r)
            w[i] = rand()
        end

        matrice_aleatoire = rand(r, r)
        S = 0.5 * (matrice_aleatoire + transpose(matrice_aleatoire))
        
        
    end 
    if init_algo=="k_means"
        # initialisation kmeans 
       
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
       
        
        for i in 1:n
            v[i]=a[i]
            w[i]= 1
        end
        
        #Normalisation des colonnes de w
        nw = zeros(1,r);
        for i = 1:n
            nw[v[i]] = nw[v[i]] + w[i]^2;
        end
        for k = 1:r
            nw[k] = sqrt(nw[k]);
        end
        for i = 1:n
            w[i] = w[i]/nw[v[i]];
        end
        #OPtimisation de S
        S=UpdateS(X,r,w,v)
          

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
        for i in 1:n
            # Trouver l'indice du premier élément non nul dans la ligne i
            idx = findfirst(x -> x != 0, W[i, :])  # Trouve le premier index où la condition est vraie
        
            if idx !== nothing
                v[i] = idx       # Stocker l'indice de l'élément non nul
                w[i] = W[i, idx] # Stocker la valeur de l'élément non nul
            else
                v[i] = 1         # Stocker l'indice de l'élément par défaut
                w[i] = W[i, 1]
            end
        end
        
        S = UpdateS(X, r, w, v)
        
        
        
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
        for i in 1:n
            # Trouver l'indice du premier élément non nul dans la ligne i
            idx = findfirst(x -> x != 0, W[i, :])  # Trouve le premier index où la condition est vraie
        
            if idx !== nothing
                v[i] = idx       # Stocker l'indice de l'élément non nul
                w[i] = W[i, idx] # Stocker la valeur de l'élément non nul
            else
                v[i] = 1         # Stocker l'indice de l'élément par défaut
                w[i] = W[i, 1]
            end
        end
        
        S = UpdateS(X, r, w, v)
        

    end 
    erreur_prec = calcul_erreur(X,S,w,v)
   
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
        w,v=UpdateW(X,S,w,v)
        S=UpdateS(X,r,w,v)
        erreur_prec = erreur
        erreur = calcul_erreur(X, S, w, v)
      
        if erreur<epsi
            break
        end
        if abs(erreur_prec-erreur)<epsi
            break
        end
    end

    
    erreur = calcul_erreur(X, S, w, v)
    
    return v,w, S, erreur
end

