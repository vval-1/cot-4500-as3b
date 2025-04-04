from main.assignment_3 import (gaussian_elimination, lu_factorization, diagonally_dominant, positive_definite)
import numpy as np

class Testing:
    def __init__(self):
        pass

    def run_tests(self):
        self.test_gaussian_elimination()
        self.test_lu_factorization()
        self.test_diagonally_dominant()
        self.test_positive_definite()

    def test_gaussian_elimination(self):
        A = np.array([[2, -1, 1],
                      [1, 3, 1],
                      [-1, 5, 4]], dtype=float)
        b = np.array([6, 0, -3], dtype=float)
        sol = gaussian_elimination(A, b)
        print("Question 1: Gaussian Elimination Solution:", sol)
        print()

    def test_lu_factorization(self):
        A = np.array([[1, 1, 0, 3],
                      [2, 1, -1, 1],
                      [3, -1, -1, 2],
                      [-1, 2, 3, -1]], dtype=float)
        L, U = lu_factorization(A)
        det = np.linalg.det(A)
        print("Question 2: LU Factorization:")
        print("Determinant:", det)
        print("L matrix:")
        print(L)
        print("U matrix:")
        print(U)
        print()

    def test_diagonally_dominant(self):
        A = np.array([[9, 0, 5, 2, 1],
                      [3, 9, 1, 2, 1],
                      [0, 1, 7, 2, 3],
                      [4, 2, 3, 12, 2],
                      [3, 2, 4, 0, 8]], dtype=float)
        res = diagonally_dominant(A)
        print("Question 3: Is the Matrix Diagonally Dominent?: ", res)
        print()

    def test_positive_definite(self):
        A = np.array([[2, 2, 1],
                      [2, 3, 0],
                      [1, 0, 2]], dtype=float)
        res= positive_definite(A)
        print("Question 4: Is the Matrix Positive Definite?: ", res)
        print()


if __name__ == "__main__":
    tester = Testing()
    tester.run_tests()
