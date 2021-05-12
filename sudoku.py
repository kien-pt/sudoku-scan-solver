import cplex
from cplex.exceptions import CplexError


def get_var_name(row, col, digit):
    return "x" + str(row * 100 + col * 10 + digit)


def lp_init(sudoku, prob):
    prob.objective.set_sense(prob.objective.sense.maximize)
    my_obj = [0] * (9 ** 3)
    my_lower_bound = [0] * (9 ** 3)
    my_upper_bound = [1] * (9 ** 3)
    my_c_type = "I" * (9 ** 3)
    my_col_names = []

    for row in range(9):
        for col in range(9):
            for digit in range(1, 10):
                varName = get_var_name(row, col, digit)
                my_col_names.append(varName)

    for row in range(9):
        for col in range(9):
            if sudoku[row][col] > 0:
                for digit in range(1, 10):
                    pos = my_col_names.index(get_var_name(row, col, digit))
                    if sudoku[row][col] == digit:
                        my_lower_bound[pos] = my_upper_bound[pos] = 1
                    else:
                        my_lower_bound[pos] = my_upper_bound[pos] = 0

    prob.variables.add(obj=my_obj, lb=my_lower_bound, ub=my_upper_bound, types=my_c_type, names=my_col_names)

    rows = []
    rhs = []

    for digit in range(1, 10):
        for col in range(9):
            variables = []
            coefficients = [1 for _ in range(9)]
            for row in range(9):
                variables.append(get_var_name(row, col, digit))
            rows.append([variables, coefficients])
            rhs.append(1)

    for digit in range(1, 10):
        for row in range(9):
            variables = []
            coefficients = [1 for _ in range(9)]
            for col in range(9):
                variables.append(get_var_name(row, col, digit))
            rows.append([variables, coefficients])
            rhs.append(1)

    for digit in range(1, 10):
        for box_top in range(3):
            for box_left in range(3):
                variables = []
                coefficients = [1 for _ in range(9)]
                for i in range(3):
                    for j in range(3):
                        row = box_top * 3 + i
                        col = box_left * 3 + j
                        variables.append(get_var_name(row, col, digit))
                rows.append([variables, coefficients])
                rhs.append(1)

    for row in range(9):
        for col in range(9):
            variables = []
            coefficients = [1 for _ in range(9)]
            for digit in range(1, 10):
                variables.append(get_var_name(row, col, digit))
            rows.append([variables, coefficients])
            rhs.append(1)

    mySense = "E" * len(rows)
    myRowNames = ["r" + str(num) for num in range(len(rows))]
    prob.linear_constraints.add(lin_expr=rows, senses=mySense, rhs=rhs, names=myRowNames)
    return my_col_names


def sudoku_solver(sudoku):
    myProb = cplex.Cplex()
    my_col_names = lp_init(sudoku, myProb)
    myProb.parameters.mip.limits.populate.set(7)

    solution = []

    try:
        myProb.populate_solution_pool()
        for i in range(myProb.solution.pool.get_num()):
            x = myProb.solution.pool.get_values(i)
            sol = [[0 for i in range(9)] for i in range(9)]
            for row in range(9):
                for col in range(9):
                    for digit in range(1, 10):
                        pos = my_col_names.index(get_var_name(row, col, digit))
                        if x[pos] == 1:
                            if sudoku[row][col] > 0:
                                sol[row][col] = -1
                            else:
                                sol[row][col] = digit
            solution.append(sol)
        return solution
    except CplexError as exc:
        print(exc)
