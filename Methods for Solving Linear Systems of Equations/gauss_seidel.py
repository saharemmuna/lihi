from numpy.linalg import norm
from matrix_utility import row_addition_elementary_matrix, scalar_multiplication_elementary_matrix, is_diagonally_dominant
import numpy as np
from colors import bcolors
from gaussian_elimination import gaussianElimination, forward_substitution, backward_substitution




def norma(mat):
    size = len(mat)
    max_row = 0
    for row in range(size):
        sum_row = 0
        for col in range(size):
            sum_row += abs(mat[row][col])
        if sum_row > max_row:
            max_row = sum_row
    return max_row






def gauss_seidel(A, b, X0, TOL=1e-16, N=200):
    n = len(A)
    k = 1

    if is_diagonally_dominant(A):
        print('Matrix is diagonally dominant - preforming gauss seidel algorithm\n')

    print("Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")
    x = np.zeros(n, dtype=np.double)
    while k <= N:

        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]

        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        if norm(x - X0, np.inf) < TOL:
            return tuple(x)

        k += 1
        X0 = x.copy()

    print("Maximum number of iterations exceeded")
    return tuple(x)


def find_u(A):
    n = A.shape[0]
    u = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1, n):
            u[i, j] = A[i, j]

    return u


def find_l(A):
    n = A.shape[0]
    l = np.zeros_like(A)

    for i in range(n):
        for j in range(i):
            l[i, j] = A[i, j]

    return l


def find_d(A):
    n = A.shape[0]
    d = np.zeros_like(A)

    for i in range(n):
        d[i, i] = A[i, i]

    return d


def add_matrices(A, B):
    size_A = A.shape
    size_B = B.shape
    rows, cols = size_A
    if not size_A == size_B:
        print("can not perform operation, sizes are not equal")
        return

    sum_mat = np.zeros_like(A)

    for i in range(rows):
        for j in range(cols):
            sum_mat[i][j] = A[i][j] + B[i][j]
    return sum_mat


def matrix_multiply(A, B):
    if len(A[0]) != len(B):
        raise ValueError("Matrix dimensions are incompatible for multiplication.")

    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    return np.array(result)


def multiply_matrix_scalar(A, scalar):
    size_A = A.shape
    rows, cols = size_A

    mult_mat = np.zeros_like(A)

    for i in range(rows):
        for j in range(cols):
            mult_mat[i][j] = A[i][j] * scalar
    return mult_mat


def MulMatrixVector(InversedMat, b_vector):
    """
    Function for multiplying a vector matrix
    :param InversedMat: Matrix nxn
    :param b_vector: Vector n
    :return: Result vector
    """
    result = []
    # Initialize the x vector
    for i in range(len(b_vector)):
        result.append([])
        result[i].append(0)
    # Multiplication of inverse matrix in the result vector
    for i in range(len(InversedMat)):
        for k in range(len(b_vector)):
            result[i][0] += InversedMat[i][k] * b_vector[k][0]
    return result



def inverse(matrix):
    #print(bcolors.OKBLUE, f"=================== Finding the inverse of a non-singular matrix using elementary row operations ===================\n {matrix}\n", bcolors.ENDC)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    n = matrix.shape[0]
    identity = np.identity(n)

    # Perform row operations to transform the input matrix into the identity matrix
    for i in range(n):
        if matrix[i, i] == 0:
            raise ValueError("Matrix is singular, cannot find its inverse.")

        if matrix[i, i] != 1:
            # Scale the current row to make the diagonal element 1
            scalar = 1.0 / matrix[i, i]
            elementary_matrix = scalar_multiplication_elementary_matrix(n, i, scalar)
            # print(f"elementary matrix to make the diagonal element 1 :\n {elementary_matrix} \n")
            matrix = np.dot(elementary_matrix, matrix)
            # print(f"The matrix after elementary operation :\n {matrix}")
            # print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------",  bcolors.ENDC)
            identity = np.dot(elementary_matrix, identity)

        # Zero out the elements above and below the diagonal
        for j in range(n):
            if i != j:
                scalar = -matrix[j, i]
                elementary_matrix = row_addition_elementary_matrix(n, j, i, scalar)
                # print(f"elementary matrix for R{j+1} = R{j+1} + ({scalar}R{i+1}):\n {elementary_matrix} \n")
                matrix = np.dot(elementary_matrix, matrix)
                # print(f"The matrix after elementary operation :\n {matrix}")
                # print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------", bcolors.ENDC)
                identity = np.dot(elementary_matrix, identity)

    return identity


def sum_vectors(a, b):
    if isinstance(a, list):
        size_A = len(a)
        size_B = len(b)
    elif isinstance(a, np.ndarray):
        size_A = a.shape[0]
        size_B = b.shape[0]
    else:
        raise TypeError("Unsupported type for 'a'. Must be a list or a NumPy array.")
    if not size_A == size_B:
        print("can not perform operation, sizes are not equal")
        return
    rows = size_A

    sum_v = np.zeros_like(a)

    for i in range(rows):
        sum_v[i][0] = a[i][0] + b[i][0]
    return sum_v


def gauss_GH(A):
    L = find_l(A)
    U = find_u(A)
    print(U)
    D = find_d(A)
    L_D = add_matrices(L, D)
    L_D_inverse = inverse(L_D)
    _L_D_inverse = multiply_matrix_scalar(L_D_inverse, -1)
    G = matrix_multiply(_L_D_inverse, U)
    print("G: ")
    print(G)

    print("H: ")
    print(L_D_inverse)






def gauss_Ait(A, Xr, b):
    L = find_l(A)
    U = find_u(A)
    D = find_d(A)
    L_D = add_matrices(L, D)
    L_D_inverse = inverse(L_D)
    _L_D_inverse = multiply_matrix_scalar(L_D_inverse, -1)
    G = matrix_multiply(_L_D_inverse, U)
    Gr = MulMatrixVector(G, Xr)
    Hr = MulMatrixVector(L_D_inverse, b)
    Xr_result=sum_vectors(Gr, Hr)

    return Xr_result

    #print("gauss")
    #print(Xr_result)


def G_norm(A):
    L = find_l(A)
    U = find_u(A)
    D = find_d(A)
    L_D = add_matrices(L, D)
    L_D_inverse = inverse(L_D)
    _L_D_inverse = multiply_matrix_scalar(L_D_inverse, -1)
    G = matrix_multiply(_L_D_inverse, U)
    #print("G matrix for Gauss_seidel: \n" + str(G))
    G_norm = norma(G)
    print("G_norm = " + str(G_norm))
    if(G_norm > 1):
        return False
    return True


if __name__ == '__main__':

    A = np.array([[2, 3, 4, 5, 6], [-5, 3, 4, -2, 3], [4, -5, -2, 2, 6], [4, 5, -1, -2, -3], [5, 5, 3, -3, 5]])
    b = np.array([70, 20, 26, -12, 37])
    X0 = np.zeros_like(b)

    A_b = [[2, 3, 4, 5, 6, 70],
           [-5, 3, 4, -2, 3, 20],
           [4, -5, -2, 2, 6, 26],
           [4, 5, -1, -2, -3, -12],
           [5, 5, 3, -3, 5, 37]]
    """"
         Date: 18/3/24
         Group: Avishag Tamssut id-326275609
                 Merav Hashta id-214718405
                 Sahar Emmuna id-213431133
         Git: https://github.com/saharemmuna/test2se.git
         Name: Sahar Emmuna id-213431133
         """

    if(G_norm(A)==0):
        result = gaussianElimination(A_b)
        print(bcolors.OKBLUE, "The norm is large 1 (the gauss seidel method will not work) the solution using the gaussian Elimination method: ", result, bcolors.ENDC)
    else:
        solution = gauss_seidel(A, b, X0)
        print(bcolors.OKBLUE,"\nApproximate solution:", solution, bcolors.ENDC)

    #--------------------------us------------------------- iterative

