import numpy as np
import math

def gaussian_elimination(mat, vec):
    n = len(vec)
    aug = np.hstack((mat, vec.reshape(-1, 1)))

    for i in range(n):
        pivot = aug[i, i]
        aug[i] = aug[i] / pivot
        for j in range(i + 1, n):
            factor = aug[j, i]
            aug[j] -= factor * aug[i]

    sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sol[i] = aug[i, -1] - np.sum(aug[i, i + 1:n] * sol[i + 1:n])

    return sol

def lu_factorization(mat):
    n = mat.shape[0]
    L = np.eye(n)
    U = mat.astype(float)

    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j] -= factor * U[i]

    return L, U

def diagonally_dominant(mat):
    n = mat.shape[0]
    for i in range(n):
        diag = abs(mat[i, i])
        off_diag = sum(abs(mat[i, j]) for j in range(n) if j != i)
        if diag < off_diag:
            return False
    return True

def positive_definite(mat):
    n = len(mat)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            total = sum(L[i][k] * L[j][k] for k in range(j))

            if i == j:
                val = mat[i][i] - total
                if val <= 0:
                    return False
                L[i][j] = math.sqrt(val)
            else:
                if L[j][j] == 0:
                    return False
                L[i][j] = (mat[i][j] - total) / L[j][j]

    return True
