using Random
using LinearAlgebra
using IterTools
using Combinatorics
using Clustering
using Hungarian 


include("ONMF.jl")

function fourth_degree_polynomial(a, b, c, d,e,x)
    return a*x^4+b*x^3+c*x^2+d*x+e
end

function roots_third_degree(a, b, c, d)
    
    if a == 0
        if b == 0
            root1 = -d/c
            return root1
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
            return [root1,root2]
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

function minimize_degre4(a,b,c,d,e)
    roots=roots_third_degree(4*a,3*b,2*c,d)
    min_x=0
    min_value=e
    for root in roots
        if root >=0
            value=fourth_degree_polynomial(a, b, c, d,e,root)
            if value<=min_value
                min_x=root
                min_value=value
            end
        end
    end 
    return min_x
end 





function calcul_erreur(X, W, S,lambda) # non relative 
    n = size(W, 2)
    result = norm(X - W * S * W')^2
    for i in 1:n
        for j in 1:n
            if i != j
                result += lambda*dot(W[:, i], W[:, j])
            end
        end
    end
    return result
end

function UpdateW(X,W,S,lambda)
    
    n,r=size(W)
    SWT=S*W'
    WSWT=W*SWT
    sumW=sum(W, dims=2)
    for p in 1:r
        
        for i in 1:n
            # retire l'ancienne valeur dans les pré calculs
            sumW[i]-=W[i,p]
            indices_sauf_i = filter(x -> x != i, 1:n)
            WSWT[i,indices_sauf_i].-=W[i,p]*(SWT[p,indices_sauf_i])
            WSWT[:,i].=WSWT[i,:]
            SWT[:,i].-=S[:,p]*W[i,p]'
            WSWT[i,i]-=2*SWT[p,i]*W[i,p]+W[i,p]^2*S[p,p]
           
           
            C=0
            D=0
            E=0
        
            for j in 1:n
                if j !=i
                    Cj=SWT[p,j]
                    Dj=WSWT[i,j]
                    E+=Cj^2
                    C+=Cj*(Dj-X[i,j])
                    D+=(X[i,j]-Dj)^2
                end 
            end 

            A=SWT[p,i]
            B=WSWT[i,i]

            a=S[p,p]^2
            b=4*A*S[p,p]
            c=4*A^2+2*S[p,p]*(B-X[i,i])+2*E
            d=4*A*(B-X[i,i])+4*C+2*lambda*(sumW[i])
            e=(X[i,i]-B)^2+2*D

            W[i,p]=minimize_degre4(a,b,c,d,e)



        
        

            # mise à jour des pré calculs :
            sumW[i]+=W[i,p]
            WSWT[i,i]+=2*SWT[p,i]*W[i,p]+W[i,p]^2*S[p,p]
            SWT[:,i].+=S[:,p]*W[i,p]'
            WSWT[i,indices_sauf_i].+=W[i,p]*(SWT[p,indices_sauf_i])
            WSWT[:,i].=WSWT[i,:]

            println("update W")
            println(calcul_erreur(X, W, S,lambda))

        end 
    end
    return W
end 

function UpdateS(X,W,S,lambda)
    WSWT=W*S*W'
    for k = 1:r
        for l = 1:r
            # supression de l'élément 
            a = 0
            b = 0
            c=0
            ind_k = findall(W[:, k] .> 0)
            ind_l = findall(W[:, l] .> 0)
            for i in ind_k
                for j in ind_l
                    WSWT[i,j]-=(W[i,k]*W[j,l]+W[i,l]*W[j,k])*S[k,l]
                    
                    a += (W[i, k] * W[j, l]+W[j,k]*W[i,l])^2
                
                    b += 2 * (W[i, k] * W[j, l]+W[j,k]*W[i,l])*(WSWT[i,j]-X[i,j])
                
                    
                    c += (X[i,j]-WSWT[i,j])^2
                    
                end
            end
            #mise à jour des pre calculs 
            if a ==0
                S[k,l]=0
            else 
                S[k, l] = min(max(-b / (2a), 0),1)
            end 
            S[l,k]=S[k,l]
            for i in ind_k
                for j in ind_l
                    WSWT[i,j]+=(W[i,k]*W[j,l]+W[i,l]*W[j,k])*S[k,l]
                end 
            end 
            println("update S")
            println(calcul_erreur(X, W, S,lambda))

        end
    end
    return S
end 



function TrisymNMF_CD(X, r,lambda, maxiter,epsi,init_algo="random",time_limit=20)
    debut = time()
    if init_algo=="random"
        # initialisation aléatoire
        n = size(X, 1)
        W = rand(n, r)
        

        matrice_aleatoire = rand(r, r)
        S = 0.5 * (matrice_aleatoire + transpose(matrice_aleatoire))
        
        
    end 
    
    erreur_prec = calcul_erreur(X, W, S,lambda)
    erreur = erreur_prec
    println(erreur)
    for itter in 1:maxiter
        # Calculer le temps écoulé
        temps_ecoule = time() - debut

        

        # Vérifier si le temps écoulé est inférieur à la limite
        if temps_ecoule > time_limit
           
            println("Limite de temps dépassée.")
            break
        end
        
        W=UpdateW(X,W,S,lambda)
        println("update W")
        println(calcul_erreur(X, W, S,lambda))
        S=UpdateS(X,W,S,lambda)
        println("update S")
        erreur_prec = erreur
        erreur = calcul_erreur(X, W, S,lambda)
        println(erreur)
        if erreur<epsi
            break
        end
        if abs(erreur_prec-erreur)<epsi
            break
        end
    end

    
    erreur = calcul_erreur(X, W, S,lambda)
    
    return W, S, erreur
end
W1=[8 0 0; 0 10 0; 0 9 0; 0 0 15]
S1=[1 0 0;0 1 0;0 0 1]
X=W1*S1*W1'
lambda=0.3
maxiter=1000
epsi=1e-2
r=3
W, S, erreur=TrisymNMF_CD(X, r,lambda, maxiter,epsi)
println(W)
println(S)
println(norm(X-W*S*W')/norm(X))
