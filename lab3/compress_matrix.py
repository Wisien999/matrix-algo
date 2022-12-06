import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt



class Node:
    def __init__(self, y_min, y_max, x_min, x_max):
        self.y_min = y_min
        self.y_max = y_max
        self.x_min = x_min
        self.x_max = x_max

        self.children = []

        self.rank = None

        self.U = None
        self.V = None

    def __str__(self):
        return 'U:' + str(self.U) + '\tV:' + str(self.V)


def truncated_svd(matrix, r):
    if r < min(matrix.shape) - 1:
        return True, svds(matrix, r + 1)
    return False, svd(matrix)


def create_tree(matrix: np.ndarray, tmin: int, tmax: int, smin: int, smax: int, r: int, ep: float) -> Node:
    temp = matrix[tmin:tmax+1, smin:smax+1]
    is_svds, (U, s, V) = truncated_svd(temp, r)

    if min(s) < ep or temp.size == 1:
        v = compress_matrix(matrix, tmin, tmax, smin, smax, U, s, V, r, is_svds)
    else:
        v = Node(tmin, tmax, smin, smax)
        tnewmax = (tmax + tmin) // 2
        snewmax = (smax + smin) // 2
        v.children.append(create_tree(matrix, tmin, tnewmax, smin, snewmax, r, ep))
        v.children.append(create_tree(matrix, tmin, tnewmax, snewmax + 1, smax, r, ep))
        v.children.append(create_tree(matrix, tnewmax + 1, tmax, smin, snewmax, r, ep))
        v.children.append(create_tree(matrix, tnewmax + 1, tmax, snewmax + 1, smax, r, ep))

    return v


def compress_matrix(matrix: np.ndarray, tmin: int, tmax: int, smin: int, smax: int,
                    U: np.ndarray, s: np.ndarray, V: np.ndarray, r: int,
                    is_svds: bool) -> Node:

    if np.max(np.abs(matrix[tmin:tmax + 1, smin:smax + 1])) < 1e-10:
        v = Node(tmin, tmax, smin, smax)
        v.rank = 0
        return v

    D = np.diag(s)
    rank = r

    v = Node(tmin, tmax, smin, smax)

    v.rank = rank
    # cut the smallest element
    if is_svds:
        v.U = U[:, 1:]
        v.V = D[1:, 1:] @ V[1:, :]
    else:
        v.U = U[:, :rank]
        v.V = D[:rank, :rank] @ V[:rank, :]

    return v


def decompress_matrix(root: Node):
    def rek(v: Node):
        for c in v.children:
            rek(c)

        if v.rank != 0 and v.rank is not None:
            matrix[v.y_min:v.y_max+1, v.x_min:v.x_max+1] = v.U @ v.V

    matrix = np.zeros((root.y_max - root.y_min + 1, root.x_max - root.x_min + 1))
    rek(root)

    return matrix


def draw(root: Node):
    def rek(v: Node):
        for c in v.children:
            rek(c)

        if v.rank != 0 and v.rank is not None:
            matrix[v.y_min, v.x_min:v.x_min+v.rank+1] = 1
            matrix[v.y_min:v.y_min+v.rank+1, v.x_min] = 1

    matrix = np.zeros((root.y_max - root.y_min + 1, root.x_max - root.x_min + 1))
    rek(root)
    plt.matshow(matrix)
    plt.show()

    return matrix


def generate_matrix(n: int, m: int, nonzero: float):
    matrix = np.random.rand(n, m)
    ys = np.random.randint(0, n-1, int((1-nonzero)*matrix.size))
    xs = np.random.randint(0, m-1, int((1-nonzero)*matrix.size))
    matrix[ys, xs] = 0

    return matrix


if __name__ == '__main__':
    N = 2**8
    m = generate_matrix(N, N, 0.01)
    print(m)
    y, x = m.shape
    print(m.shape)
    r = create_tree(m, 0, y - 1, 0, x - 1, 2 ** 2, np.float32(1e-5))
    print(np.allclose(decompress_matrix(r), m))
    print(draw(r))
