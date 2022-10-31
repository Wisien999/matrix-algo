import numpy as np
import sys
import time

ADD_CNT = 0
MULT_CNT = 0
TEST_RANGE = 10


def binet(X: np.matrix, Y: np.matrix):
    if X.shape == Y.shape == (2, 2):
        return multiply_2x2(X, Y)
    else:
        _size, _ = X.shape
        # lewy górny róg macierzy
        X11 = X[0:_size//2, 0:_size//2]
        Y11 = Y[0:_size//2, 0:_size//2]
        # prawy górny róg macierzy
        X12 = X[0:_size//2, _size//2:_size]
        Y12 = Y[0:_size//2, _size//2:_size]
        # lewy dolny róg macierzy
        X21 = X[_size//2:_size, 0:_size//2]
        Y21 = Y[_size//2:_size, 0:_size//2]
        # prawy dolny róg macierzy
        X22 = X[_size//2:_size, _size//2:_size]
        Y22 = Y[_size//2:_size, _size//2:_size]

        # rekurencyjne obliczanie czterech ćwiartek macierzy będącej
        # wynikiem mnozenia macierzy X i Y
        R11 = binet(X11, Y11) + binet(X12, Y21)
        R12 = binet(X11, Y12) + binet(X12, Y22)
        R21 = binet(X21, Y11) + binet(X22, Y21)
        R22 = binet(X21, Y12) + binet(X22, Y22)

        # łączenie ćwiartek w wynikową macierz
        return np.concatenate((
            np.concatenate((R11, R12), axis=1),
            np.concatenate((R21, R22), axis=1)
        ),
            axis=0)


def multiply_2x2(X: np.matrix, Y: np.matrix):
    global ADD_CNT, MULT_CNT
    ADD_CNT += 4
    MULT_CNT += 8
    return np.matrix([[X[0, 0] * Y[0, 0] + X[0, 1] * Y[1, 0],
                       X[0, 0] * Y[0, 1] + X[0, 1] * Y[1, 1]],
                      [X[1, 0] * Y[0, 0] + X[1, 1] * Y[1, 0],
                       X[1, 0] * Y[0, 1] + X[1, 1] * Y[1, 1]]])


if __name__ == '__main__':
    for i in range(1, TEST_RANGE + 1):
        # generujemy macierze o wymiarach 2^1, 2^2 ... 2^TEST_RANGE
        n = 2 ** i

        M1 = np.random.rand(n, n)
        M2 = np.random.rand(n, n)

        start_time = time.time()

        RESULT = binet(M1, M2)

        end_time = time.time()

        print(
            "Obliczenia dla macierzy kwadratowej o szerokości i długości {dim} zajęły {time:.5f} sekund".format(dim=n, time=end_time - start_time))
        print("Wykonano {mult} mnożeń i {add} dodawań.".format(mult=MULT_CNT, add=ADD_CNT))
