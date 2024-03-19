import numpy as np
from colors import bcolors
from matrix_utility import swap_row

def create_diagonal_matrix(values):
    """
    יוצרת מטריצה מדורגת באלכסון עם הערכים שנמצאים ברשימה values על האלכסון הראשי.

    :param values: רשימה של הערכים שיש להניח על האלכסון הראשי.
    :return: מטריצה מדורגת.
    """
    n = len(values)
    # יצירת מטריצה ריקה בגודל n על n
    matrix = np.zeros((n, n))
    # הוספת הערכים לאלכסון הראשי של המטריצה
    for i in range(n):
        matrix[i, i] = values[i]
    return matrix


def gaussianElimination(mat):
    N = len(mat)

    singular_flag = forward_substitution(mat)

    if singular_flag != -1:

        if mat[singular_flag][N]:
            return "Singular Matrix (Inconsistent System)"
        else:
            return "Singular Matrix (May have infinitely many solutions)"

    # if matrix is non-singular: get solution to system using backward substitution
    return backward_substitution(mat)



def forward_substitution(mat):
    N = len(mat)
    for k in range(N):

        # Partial Pivoting: Find the pivot row with the largest absolute value in the current column
        pivot_row = k
        v_max = mat[pivot_row][k]
        for i in range(k + 1, N):
            if abs(mat[i][k]) > v_max:
                v_max = mat[i][k]
                pivot_row = i

        # if a principal diagonal element is zero,it denotes that matrix is singular,
        # and will lead to a division-by-zero later.
        if not mat[k][pivot_row]:
            return k  # Matrix is singular

        # Swap the current row with the pivot row
        if pivot_row != k:
            swap_row(mat, k, pivot_row)
        # End Partial Pivoting

        for i in range(k + 1, N):

            #  Compute the multiplier
            m = mat[i][k] / mat[k][k]

            # subtract fth multiple of corresponding kth row element
            for j in range(k + 1, N + 1):
                mat[i][j] -= mat[k][j] * m

            # filling lower triangular matrix with zeros
            mat[i][k] = 0

    return -1


# function to calculate the values of the unknowns
def backward_substitution(mat):
    N = len(mat)
    x = np.zeros(N)  # An array to store solution

    # Start calculating from last equation up to the first
    for i in range(N - 1, -1, -1):

        x[i] = mat[i][N]

        # Initialize j to i+1 since matrix is upper triangular
        for j in range(i + 1, N):
            x[i] -= mat[i][j] * x[j]

        x[i] = (x[i] / mat[i][i])

    return x


if __name__ == '__main__':
    """"
           Date: 18/3/24
           Group: Avishag Tamssut id-326275609
                   Merav Hashta id-214718405
                   Sahar Emmuna id-213431133
           Git: https://github.com/Avishagtams/Numerical-Analysis-Quiz2.git
           Name: Avishag Tamssut 326275609
           """
    A_b = [[2, 3, 4, 5, 6, 70],
        [-5, 3, 4, -2, 3, 20],
        [4, -5, -2, 2, 6, 26],
        [4, 5, -1, -2, -3, -12],
        [5, 5, 3, -3, 5, 37]]

    result = gaussianElimination(A_b)
    if isinstance(result, str):
        print(result)
    else:
        print(bcolors.OKBLUE,"\nSolution for the system:")
        for x in result:
            print("{:.6f}".format(x))