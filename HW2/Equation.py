import numpy as np


def jacobi(A: np.matrix, b: np.array, x_0: np.array, n: int):
    U, D, L = np.triu(A,1), np.tril(np.triu(A)), np.tril(A, -1)
    counter = 0
    D_inverse = np.linalg.inv(D)
    M = L + U
    x_i = x_0
    convergence_matrix = D_inverse@M
    while(counter < n):
        x_iplus = D_inverse@(b-M@x_i)
        print(f"x_{counter+1}: ", x_iplus)
        x_i = x_iplus
        counter += 1
    return x_i, convergence_matrix

def gauss_seidel(A: np.matrix, b: np.array, x_0: np.array, n: int):
    U, D, L = np.triu(A,1), np.tril(np.triu(A)), np.tril(A, -1)    
    counter = 0
    M = L + D
    M_inverse = np.linalg.inv(M)
    # x_(k+1) = M-1(b-Ux_(k))
    x_i = x_0
    convergence_matrix = M_inverse@U
    while(counter < n):
        x_iplus = M_inverse@(b-U@x_i)
        print(f"x_{counter+1}: ", x_iplus)
        x_i = x_iplus
        counter += 1
   
    return x_i, convergence_matrix