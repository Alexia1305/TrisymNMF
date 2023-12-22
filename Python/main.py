import numpy as np


def min_degre4(a, b, c):
    # minimum du polynome de degré 4 ax4+bx2+cx
    min = 0
    sol = 0
    if a==0:
        x=-c/(2*b)
        value = a * (x ** 4) + b * (x ** 2) + c * x
        if x > 0 and value < min:
            sol = x
            min = value
        return sol,min
    # méthode de cardan
    p = 2 * b / (4 * a)
    q = c / (4 * a)
    delta = -(4 * (p ** 3) + 27 * (q ** 2))
    if delta < 0:
        x = np.cbrt((-q + (-delta / 27) ** (1 / 2))/2) + np.cbrt((-q - (-delta / 27) ** (1 / 2))/2)
        value = a * (x ** 4) + b * (x ** 2) + c * x
        if x > 0 and value < min:
            sol = x
            min = value
    elif delta == 0:
        if p == q and q == 0:
            x = 0
            value = 0
        else:
            x = 3 * q / p
            value = a * (x ** 4) + b * (x ** 2) + c * x
            if x > 0 and value < min:
                sol = x
                min = value
            x = -3 * q / (2 * p)
            value = a * (x ** 4) + b * (x ** 2) + c * x
            if x > 0 and value < min:
                sol = x
                min = value
    else:
        for k in range(3):
            x = 2 * (-p / 3) ** (1 / 2) * np.cos(
                (1 / 3) * np.arccos((3 * q / (2 * p)) * ((3 / -p) ** (1 / 2))) + (2 * k * np.pi / 3))
            value = a * (x ** 4) + b * (x ** 2) + c * x
            if x > 0 and value < min:
                sol = x
                min = value
    return sol, min


def calcul_erreur(X, W, S):
    # Calcul de X - WSW^T
    result_matrix = X - np.dot(np.dot(W, S), W.T)

    # Calcul de la norme de Frobenius au carré
    erreur = np.sum(result_matrix ** 2) / np.sum(X ** 2)
    return erreur


def symTriONMF_coordinate_descent(X,r, maxiter):
    # initialisation aléatoire
    n= X.shape[0]
    W = np.zeros((n, r))
    for i in range(n):
        k = np.random.randint(r)
        W[i, k] = np.random.rand()
    matrice_aleatoire = np.random.rand(r, r)
    S = 0.5 * (matrice_aleatoire + matrice_aleatoire.T)
    erreur_prec = calcul_erreur(X, W, S)
    erreur = erreur_prec
    print(erreur_prec)

    for itter in range(maxiter):
        # optimisation de W

        for i in range(n):
            k_min = None
            k_min_value = None
            sum_value = float('inf')
            for k in range(r):
                a = S[k, k] ** 2
                b = -2 * X[i, i] * S[k, k]
                c = 0
                for j in range(n):
                    if (j == i):
                        continue
                    l = np.argmax(W[j, :])  # l'élément non nul
                    b += 2*(S[k, l] * W[j, l]) ** 2
                    c += -4 * X[i, j] * S[k, l] * W[j, l]
                x, value_x = min_degre4(a, b, c)
                if value_x < sum_value:
                    k_min = k
                    k_min_value = x
                    sum_value = value_x
            W[i, :] = np.zeros(r)
            W[i, k_min] = k_min_value
        erreur_prec = erreur
        erreur = calcul_erreur(X, W, S)
        print("W "+str(erreur))

        # optimisation de S

        for k in range(r):
            for l in range(r):  # symétrique
                a = 0
                b = 0
                ind_i = np.argwhere(W[:, k] > 0)
                if np.size(ind_i) == 0:
                    break
                ind_j = np.argwhere(W[:, l] > 0)
                if np.size(ind_j) == 0:
                    break
                for i in ind_i:
                    for j in ind_j:
                        a += (W[i, k] * W[j, l]) ** 2
                        b += 2 * X[i, j] * W[i, k] * W[j, l]
                S[k, l] = max(b / (2 * a),0)


        erreur_prec = erreur
        erreur = calcul_erreur(X, W, S)
        print("error_S "+str(erreur))


    return W, S, erreur


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # création matrice
    n = 20
    r = 5
    W = np.zeros((n, r))
    for i in range(n):
        k = np.random.randint(r)
        W[i, k] = np.random.rand()
    matrice_aleatoire = np.random.rand(r, r)
    S = 0.5 * (matrice_aleatoire + matrice_aleatoire.T)

    X = np.dot(W, np.dot(S, np.transpose(W)))
    print(X)

    # algorithme :
    W, S, erreur = symTriONMF_coordinate_descent(X,r, maxiter=1000)
    print(X)
    print(W)
    print(S)
    print(np.dot(W,np.dot(S,W.T)))
