
function Modularity(A,partition)
    n,_=size(A)
    m = count(x -> x != 0, A)/2
    
    k = sum(A, dims=2)

    
    k = vec(k)  
    Q=0
    for i in 1:n 
        for j in 1:n 
            if partition[i]==partition[j]
                Q+= A[i,j]-(k[i]*k[j]/(2*m))
            end 
        end 
    end
   
    return Q / (2 * m)
end
function confusion_matrix(A, B)
    # Trouver les labels uniques dans chaque partition
    A_labels = length(unique(A))
    B_labels = length(unique(B))

    # Initialiser la matrice de confusion
    M = zeros(Int, length(A_labels), length(B_labels))

    # Remplir la matrice de confusion
    for (i, a) in enumerate(A_labels)
        for (j, b) in enumerate(B_labels)
            M[i, j] = sum((A .== a) .& (B .== b))
        end
    end

    return M
end

function normalised_mutual_info(A,B)
    # A the real partition 
    # B is the found partition
    n=length(A)
    M=confusion_matrix(A, B)
    ca,cb=size(M)
   
    sum1=0
    sum2=0
    sum3=0
    Mi= sum(M, dims=2)  # Somme des lignes (clusters de A)
    Mj = sum(M, dims=1)  # Somme des colonnes (clusters de B)
    for i in 1:ca
        
        sum2+=Mi[i]*log(Mi[i]/n)
        for j in 1:cb 
            sum1+=-2*M[i,j]*log(M[i,j]*n/(Mi[i]*Mj[j]))
            sum3+=Mj[j]*log(Mj[j]/n)
        end 
    end 
    
    return sum1/(sum2+sum3)

end 
function read_pajek(filename::String)
    # Lire tout le fichier
    lines = readlines(filename)
    
    # Trouver la section des sommets
    vertex_index = findfirst(x -> startswith(x, "*Vertices"), lines)
    if vertex_index == nothing 
        vertex_index = findfirst(x -> startswith(x, "*vertices"), lines)

    end 
    num_vertices = parse(Int, split(lines[vertex_index])[2])
    label_vertices=fill("", num_vertices)
    for node in 1:num_vertices
        node_data=split(lines[vertex_index+node], r"\s+", keepempty=false)
        label_vertices[node]=strip(node_data[2], ['"'])
    end 
    
    # Trouver la section des arêtes
    edges_index = findfirst(x -> startswith(x, "*Edges"), lines)
    if edges_index == nothing 
        edges_index = findfirst(x -> startswith(x, "*edges"), lines)

    end 
    # Créer un graphe avec le nombre de sommets trouvé
    g = SimpleGraph(num_vertices)
    
    # Ajouter les arêtes à partir des lignes qui suivent "*Edges"
    for line in lines[(edges_index+1):end]
        edge = parse.(Int, split(line))
        add_edge!(g, edge[1], edge[2])
    end
    
    return g,label_vertices
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

