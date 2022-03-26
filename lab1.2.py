import numpy as np

def get_matrix_ABC(matrix):
    n = matrix.shape[0]
    res_matrix = np.zeros((n,3))
    res_matrix[0][1] = matrix[0][0]
    res_matrix[0][2] = matrix[0][1]
    for i in range(1, n - 1):
        res_matrix[i][0] = matrix[i][i-1]
        res_matrix[i][1] = matrix[i][i]
        res_matrix[i][2] = matrix[i][i+1]
    res_matrix[n-1][0] = matrix[n-1][n-2]
    res_matrix[n-1][1] = matrix[n-1][n-1]

    return res_matrix

def TDMA(matrix, matrix_d):
    n = matrix.shape[0]
    matrix_p = np.zeros((n,1))
    matrix_q = np.zeros((n,1))
    matrix_p[0, 0] = -matrix[0, 2]/matrix[0,1]
    matrix_q[0, 0] = matrix_d[0, 0]/matrix[0,1]
    for i in range(1, n):
        matrix_p[i,0] = -(matrix[i,2])/(matrix[i,1] + matrix[i,0]*matrix_p[i-1,0])
        matrix_q[i,0] = (matrix_d[i,0] - matrix[i,0]*matrix_q[i-1,0])/(matrix[i,1] + matrix[i,0]*matrix_p[i-1,0])


    matrix_x = np.zeros((n,1))
    matrix_x[n-1,0] = matrix_q[n-1,0]
    for i in range(n-2, -1, -1):
        matrix_x[i,0] = matrix_p[i,0]*matrix_x[i+1,0] + matrix_q[i,0]

    return matrix_x

if __name__ == "__main__":
    matrix = np.array( [
        [8, -2, 0, 0, 0],
        [7, -19, 9, 0, 0],
        [0, -4, 21, -8, 0],
        [0, 0, 7, -23, 9],
        [0, 0, 0, 4, -7],
    ], dtype = 'float')
    matrix_abc = get_matrix_ABC(matrix);
    matrix_d = np.array([
        [-14],
        [-55],
        [49],
        [86],
        [8]
    ])
    print(TDMA(matrix_abc, matrix_d))
