import numpy as np


def get_matrix_alp(matrix_a):
    matrix_alp = np.zeros(matrix_a.shape)
    n = matrix_a.shape[0] # Number rows and columns
    for i in range(n):
        for j in range(n):
            if (i == j):
                continue
            matrix_alp[i, j] = -matrix_a[i, j]/matrix_a[i, i]

    return matrix_alp


def get_matrix_bet(matrix_a, matrix_b):
    matrix_bet = np.zeros(matrix_b.shape)
    n = matrix_a.shape[0] # Number rows and columns
    for i in range(n):
        matrix_bet[i, 0] = matrix_b[i, 0] / matrix_a[i, i]

    return matrix_bet


def get_norm(original_matrix):
    matrix = original_matrix.copy()
    n = matrix.shape[0] # Number rows and columns
    if (matrix.shape[1] == 1):
        return max(map(abs,matrix[:,0]))
    for i in range(n):
        matrix[i, 0] = abs(matrix[i, 0])
        for j in range(1, len(matrix[i])):
            matrix[i, 0] += abs(matrix[i, j])


    return max(matrix[:,0])


def method_jacobi(matrix_alp, matrix_bet, eps):
    n = matrix_alp.shape[0]
    prev_x = np.zeros((n, 1))
    cur_x = np.matmul(matrix_alp, prev_x) + matrix_bet
    alp_norm = get_norm(matrix_alp)
    if alp_norm >= 1:
        while(factor*get_norm(cur_x - prev_x) > eps and q < 100):
            prev_x = cur_x
            cur_x = np.matmul(matrix_alp, prev_x) + matrix_bet
            q += 1

        return cur_x, q
    factor = alp_norm / (1 - alp_norm)
    q = 0
    while(factor*get_norm(cur_x - prev_x) > eps):
        prev_x = cur_x
        cur_x = np.matmul(matrix_alp, prev_x) + matrix_bet
        q += 1
    
    return cur_x, q


def new_mut(matrix_alp, matrix_bet, matrix_x):
    n = matrix_alp.shape[0]
    new_x = np.zeros((n, 1))
    for i in range(n):
        for j in range(i):
            new_x[i, 0] += matrix_alp[i, j]*new_x[j, 0] 
        for j in range(i, n):
            new_x[i, 0] += matrix_alp[i, j]*matrix_x[j, 0] 

        new_x[i, 0] += matrix_bet[i, 0]

    return new_x



def method_seidel(matrix_alp, matrix_bet, eps):
    n = matrix_alp.shape[0]
    prev_x = np.zeros((n, 1))
    cur_x = new_mut(matrix_alp, matrix_bet, prev_x)
    matrix_alp_1 = np.tril(matrix_alp, -1)
    matrix_alp_2 = np.triu(matrix_alp)
    if (get_norm(matrix_alp) >= 1):
        while(get_norm(cur_x - prev_x) > eps and q < 100):
            prev_x = cur_x
            cur_x = new_mut(matrix_alp, matrix_bet, prev_x)
            q += 1

        return cur_x, q

    factor = get_norm(matrix_alp_2)/ (1 - get_norm(matrix_alp))
    q = 0
    while(factor*get_norm(cur_x - prev_x) > eps):
        prev_x = cur_x
        cur_x = new_mut(matrix_alp, matrix_bet, prev_x)
        q += 1

    return cur_x, q


def main():
    matrix_a = np.array([
        [18, 8, -3, 4],
        [-7, 15, -5, -2],
        [-4, 0, 13, 4],
        [-8, -8, -6, 31],
    ])
    matrix_b = np.array([
        [-84],
        [-5],
        [-38],
        [263]
    ])
    n = matrix_a.shape[0] # Number rows and columns
    matrix_alp = get_matrix_alp(matrix_a)
    matrix_bet = get_matrix_bet(matrix_a, matrix_b)
    eps = float(input())

    solve, q = method_jacobi(matrix_alp, matrix_bet, eps)
    print("Method Jacobi solving: ")
    print(solve)
    print(f"Number of iterations: {q}")

    solve, q = method_seidel(matrix_alp, matrix_bet, eps)
    print("Method Seidel solving: ")
    print(solve)
    print(f"Number of iterations: {q}")


if __name__ == "__main__":
    main()
