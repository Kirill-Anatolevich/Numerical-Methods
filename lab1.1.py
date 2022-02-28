import numpy as np
from scipy.linalg import lu


def set_max_value(cur_pos, matrix_l, matrix_a, matrix_b):
    """Function find max value in column and swaps rows with max value and current"""
    cur_column = matrix_a[range(cur_pos,matrix_l.shape[0]),cur_pos]
    max_value = np.argmax(np.absolute(cur_column))
    if cur_pos != cur_pos + max_value:
        matrix_a[[cur_pos, cur_pos+max_value]] = matrix_a[[cur_pos+max_value, 
                                                                 cur_pos]]
        matrix_b[[cur_pos, cur_pos+max_value]] = matrix_b[[cur_pos+max_value, 
                                                                    cur_pos]]
        if cur_pos != 0:
            matrix_l[[cur_pos, cur_pos+max_value]] = matrix_l[[cur_pos+max_value, 
                                                                     cur_pos]]
            matrix_l[:,[cur_pos, cur_pos+max_value]] = matrix_l[:,[cur_pos+max_value, 
                                                                   cur_pos]]

def get_matrix_lu(matrix_a, matrix_b):
    """Function returns matrix L, matrix U and new matrix B
        AX = B
        A = LU"""
    size_a = matrix_a.shape[0]
    size_b = matrix_b.shape[1] # Number of columns
    matrix_l = np.identity(size_a, dtype = 'float')
    for k in range(size_a - 1):
        set_max_value(k, matrix_l, matrix_a, matrix_b)
        for i in range(k + 1, size_a):
            h = matrix_a[i,k]/matrix_a[k,k]
            matrix_l[i, k] = h
            for j in range(size_a):
                matrix_a[i,j] -= h*matrix_a[k,j]

    return matrix_l, matrix_a, matrix_b


def get_solve(matrix_a, matrix_b):
    """Function returns matrix X which is solve of systems of equations"""
    matrix_l, matrix_u, matrix_b = get_matrix_lu(np.copy(matrix_a), matrix_b)
    size_l = matrix_l.shape[0]
    matrix_z = np.empty((size_l, 1))
    for i in range(size_l):
        matrix_z[i,0] = matrix_b[i,0]
        for j in range(0, i):
            matrix_z[i,0] -= matrix_l[i,j]*matrix_z[j,0]



    matrix_x = np.empty((size_l, 1))
    for i in range(size_l-1, -1, -1):
        matrix_x[i,0] = matrix_z[i,0]
        for j in range(i+1, size_l):
            matrix_x[i,0] -= matrix_u[i,j]*matrix_x[j,0]
        matrix_x[i,0] /= matrix_u[i,i]

    return matrix_x


def get_inverse_matrix(matrix):
    matrix_b = np.identity(matrix.shape[0], dtype = 'float')
    result = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        result[:,[i]] += get_solve(np.copy(matrix), matrix_b[:,[i]])

    return result

def get_det(matrix):
    matrix_b = np.eye(matrix.shape[0],1)
    matrix_l, matrix_u, matrix_b = get_matrix_lu(np.copy(matrix), matrix_b)
    res = 1
    for i in range(matrix.shape[0]):
        res*= matrix_u[i][i]
    return res

def lab1_1():
    matrix_a = np.array([
        [1, -1, 1, -2],
        [-9, -1, 1, 8],
        [-7, 0, 8, -6],
        [3, -5, 1, -6],
    ], dtype = 'float')
    matrix_b = np.array([
        [-20],
        [60],
        [-60],
        [-44],
    ], dtype = 'float')
    matrix_l, matrix_u, matrix_b_aft = get_matrix_lu(np.copy(matrix_a), 
                                                 np.copy(matrix_b))
    p, l, u = lu(np.copy(matrix_a))
    print("My matrix L:")
    print(matrix_l)
    print("Scipy matrix L:")
    print(l)
    print("My matrix U:")
    print(matrix_u)
    print("Scipy matrix U:")
    print(u)
    print("My determinant:")
    print(get_det(np.copy(matrix_a)))
    print("Numpy determinant:")
    print(np.linalg.det(np.copy(matrix_a)))
    print("My matrix X:")
    print(get_solve(np.copy(matrix_a), np.copy(matrix_b)))
    print("Numpy matirx X:")
    print(np.linalg.solve(np.copy(matrix_a), np.copy(matrix_b)))
    print("My inverse matrix:")
    print(get_inverse_matrix(np.copy(matrix_a)))
    print("Numpy inverse matrix:")
    print(np.linalg.inv(matrix_a))


def main():
    lab1_1()

if __name__ == '__main__':
    main()
