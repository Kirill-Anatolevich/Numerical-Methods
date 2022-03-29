import numpy as np
import math


def find_max(matrix):
    n = matrix.shape[0]
    max_elem = -2e9
    max_i = -1
    max_j = -1
    for i in range(n-1):
        j = np.argmax(matrix[i, (i+1):n])
        # print(matrix[i, (i+1):n])
        if (max_elem < matrix[i, i + j + 1]):
            max_i = i
            max_j = i + j + 1
            max_elem = matrix[max_i, max_j]

    return max_i, max_j


def get_orthohonal(i, j, matrix):
    orthogonal = np.eye(matrix.shape[0])
    if (matrix[i,i]-matrix[j,j] == 0):
        angle = math.pi/4
    else:
        angle = 1/2*(math.atan(2*matrix[i,j]/(matrix[i,i]-matrix[j,j])))
    orthogonal[i,i] = math.cos(angle)
    orthogonal[i,j] = -math.sin(angle)
    orthogonal[j,i] = math.sin(angle)
    orthogonal[j,j] = math.cos(angle)

    return orthogonal


def check_diag(matrix):
    res = 0
    """
    for i in range(1, matrix.shape[0]):
        res += sum(map(lambda x: x**2, matrix[i, :i]))
    """
    for i in range(1, matrix.shape[0]):
        for j in range(i):
            res += matrix[i,j]**2

    # print(res)
    return res**(0.5)


def rotation_method(matrix_a, eps):
    i,j = find_max(matrix_a)
    k = 0
    eigenvectors = np.eye(matrix_a.shape[0])
    while (check_diag(matrix_a) > eps):
        i,j = find_max(matrix_a)
        cur_u = get_orthohonal(i, j, matrix_a)
        matrix_a = np.matmul(np.matmul(cur_u.transpose(),matrix_a), cur_u)
        eigenvectors = np.matmul(eigenvectors, cur_u)

    eigenvalues = np.asarray([matrix_a[i,i] for i in range(matrix_a.shape[0])])
    return eigenvectors, eigenvalues



def main():
    matrix_a = np.array([
        [4, 2, 1],
        [2, 5, 3],
        [1, 3, 6]
    ])
    matrix_a = np.array([
        [2, 8, 7],
        [8, 2, 7],
        [7, 7, -8]
    ])

    eps = float(input())
    eigenvectors, eigenvalues = rotation_method(matrix_a, eps)
    print(eigenvectors)
    print("x1*x2: ")
    print(np.dot(eigenvectors[:,0], eigenvectors[:,1]))
    print("x2*x3: ")
    print(np.dot(eigenvectors[:,1], eigenvectors[:,2]))
    print("x1*x3: ")
    # print(eigenvectors[:,0], eigenvectors[:,2])
    print(np.dot(eigenvectors[:,0], eigenvectors[:,2]))




if __name__ == "__main__":
    main()
