from mip import *


def solution(table):
    # init mlp model
    m = Model()

    # init model variables
    matrix = [[[m.add_var(var_type=BINARY) for digit in range(9)] for column in range(9)] for row in range(9)]

    for row in range(9):
        for col in range(9):
            if table[row][col] > 0:
                m += matrix[row][col][int(table[row][col]) - 1] == 1

    for row in range(9):
        for col in range(9):
            m += xsum(matrix[row][col][digit] for digit in range(9)) == 1

    for row in range(9):
        for digit in range(9):
            m += xsum(matrix[row][col][digit] for col in range(9)) == 1

    for col in range(9):
        for digit in range(9):
            m += xsum(matrix[row][col][digit] for row in range(9)) == 1

    for box_top in range(0, 7, 3):
        for box_left in range(0, 7, 3):
            for digit in range(9):
                m += xsum(matrix[box_top + (i // 3)][box_left + (i % 3)][digit] for i in range(9)) == 1

    m.emphasis = 1  # search for feasible first
    m.max_solutions = 10  # stop when a solution is found
    m.max_seconds = 500
    m.optimize()


    for row in range(9):
        for col in range(9):
            for digit in range(9):
                if matrix[row][col][digit].x == 1:
                    if table[row][col] > 0:
                        table[row][col] = -1
                    else:
                        table[row][col] = digit + 1

    return table

table = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# table = [
#     [9, 2, 6, 5, 7, 1, 4, 8, 3],
#     [3, 5, 1, 4, 8, 6, 2, 7, 9],
#     [8, 7, 4, 9, 2, 3, 5, 1, 6],
#     [5, 8, 2, 3, 6, 7, 1, 9, 4],
#     [1, 4, 9, 2, 5, 8, 3, 6, 7],
#     [7, 6, 3, 1, 0, 0, 8, 2, 5],
#     [2, 3, 8, 7, 0, 0, 6, 5, 1],
#     [6, 1, 7, 8, 3, 5, 9, 4, 2],
#     [4, 9, 5, 6, 1, 2, 7, 3, 8],
# ]

solution(table)