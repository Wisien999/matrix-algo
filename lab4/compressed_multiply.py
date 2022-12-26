import numpy as np
from compress_matrix import generate_matrix, create_tree, decompress_matrix, Node
from copy import deepcopy


# Multiply compressed matrix by given vector X
def matrix_vector_mult(M: Node, X : np.ndarray):
    n, m = X.shape
    if len(M.children) == 0:
        if M.rank == 0:
            return np.zeros((n, n))
        else:
            return M.U @ (M.V @ X)
    else:
        X1, X2 = X[0:n//2, 0:1], X[n//2:n, 0:1]
        Y11 = matrix_vector_mult(M.children[0], X1)
        Y12 = matrix_vector_mult(M.children[1], X2)
        Y21 = matrix_vector_mult(M.children[2], X1)
        Y22 = matrix_vector_mult(M.children[3], X2)

        result = np.concatenate((Y11 + Y12, Y21 + Y22), axis = 0)

        return result

# Multiply compressed matrix by itself
def matrix_mult(M: Node):
    n, m = M.y_max - M.y_min + 1, M.x_max - M.x_min + 1 
    if len(M.children) == 0:
        if M.rank == 0:
            return np.zeros((n, m))
        else:
            return M.U @ (M.V @ M.U) @ M.V
    else:
        A = decompress_matrix(M)
        A11 = A[0: n//2, 0: m//2]
        A12 = A[0: n//2, m//2 : m]
        A21 = A[n//2 : n, 0: m//2]
        A22 = A[n//2 : n, m//2 : m]

        R11 = np.add((A11 @ A11), (A12 @ A21))
        R12 = np.add((A11 @ A12), (A12 @ A22))
        R21 = np.add((A21 @ A11), (A22 @ A21))
        R22 = np.add((A21 @ A12), (A22 @ A22))

        result = np.concatenate(
            (
                np.concatenate((R11, R12), axis = 1),
                np.concatenate((R21, R22), axis = 1)
            ), 
            axis = 0
        )
        return result


if __name__ == '__main__':
    N = 2

    M = generate_matrix(2 ** N, 2 ** N, 0.8)
    print('Matrix:')
    print(M)
    tree_M = create_tree(M, 0, 2**N - 1, 0, 2**N -1, 2**2, np.float32(1e-5))

    print('Normal multiply:')
    print(M @ M)

    print('My multiply:')
    print(matrix_mult(tree_M))
    
