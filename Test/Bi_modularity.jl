
function BI_Modularity(A,delta)
    n,_=size(A)
    m = count(x -> x != 0, A)
    
    k = sum(A, dims=2)

    
    k = vec(k)  
    Q=0
    for i in 1:n 
        for j in 1:n 
            if delta[i,j] !=0
                Q+= A[i,j]-(k[i]*k[j]/m)
            end 
        end 
    end
    Q/=2*m
    return Q 
end

# A = [
#     1 1 1 1 1 1 0 1 1 0 0 0 0 0;
#     1 1 1 0 1 1 1 1 0 0 0 0 0 0;
#     0 1 1 1 1 1 1 1 1 0 0 0 0 0;
#     1 0 1 1 1 1 1 1 0 0 0 0 0 0;
#     0 0 1 1 1 0 1 0 0 0 0 0 0 0;
#     0 0 1 0 1 1 0 1 0 0 0 0 0 0;
#     0 0 0 0 1 1 1 1 0 0 0 0 0 0;
#     0 0 0 0 0 1 0 1 1 0 0 0 0 0;
#     0 0 0 0 1 0 1 1 1 0 0 0 0 0;
#     0 0 0 0 0 0 1 1 1 0 0 1 0 0;
#     0 0 0 0 0 0 0 1 1 1 0 1 0 0;
#     0 0 0 0 0 0 0 1 1 1 0 1 1 1;
#     0 0 0 0 0 0 1 1 1 1 0 1 1 1;
#     0 0 0 0 0 1 1 0 1 1 1 1 1 1;
#     0 0 0 0 0 0 1 1 0 1 1 1 0 0;
#     0 0 0 0 0 0 0 1 1 0 0 0 0 0;
#     0 0 0 0 0 0 0 0 1 0 1 0 0 0;
#     0 0 0 0 0 0 0 0 1 0 1 0 0 0
# ]

# p,q=size(A)
# # Afficher la matrice
# println(A)
# n=p+q
# X=zeros(n,n)
# for i in 1:p
#     for j in 1:q
#         if A[i,j]==1
#             X[i,p+j]=1
#             X[p+j,i]=1
#         end
#     end
# end
# clusters=[[19,20,21,22,23,24],[25,26],[28,30,31,32],[27,29],[1,2,3,4,5,6],[7,9,10],[8,16,17,18],[11,12,13,14,15]]
# delta=zeros(n,n)
# for c in clusters
#     groups_of_2 = collect(combinations(c, 2))
#     for (i,j) in groups_of_2
#         delta[i,j] =1
#     end 
# end

# print(BI_Modularity(X,delta))

