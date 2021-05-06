import cplex
from cplex.exceptions import CplexError


def sudokuReader(filename):
    sudoku = []
    with open(filename) as f:
        for line in f.readlines():
            row = [int(x) for x in line.split()]
            sudoku.append(row)
    return sudoku


def getVarName(row, col, digit):
    return "x" + str(row * 100 + col * 10 + digit)


def lpInitialization(sudoku, prob):
    prob.objective.set_sense(prob.objective.sense.maximize)
    myObj = [0] * (9 ** 3)
    myLowerBound = [0] * (9 ** 3)
    myUpperBound = [1] * (9 ** 3)
    myCtype = "I" * (9 ** 3)
    myColNames = []

    for row in range(9):
        for col in range(9):
            for digit in range(1, 10):
                varName = getVarName(row, col, digit)
                myColNames.append(varName)

    for row in range(9):
        for col in range(9):
            if sudoku[row][col] > 0:
                for digit in range(1, 10):
                    pos = myColNames.index(getVarName(row, col, digit))
                    if sudoku[row][col] == digit:
                        myLowerBound[pos] = myUpperBound[pos] = 1
                    else:
                        myLowerBound[pos] = myUpperBound[pos] = 0

    prob.variables.add(obj=myObj, lb=myLowerBound, ub=myUpperBound, types=myCtype, names=myColNames)

    rows = []
    rhs = []

    for digit in range(1, 10):
        for col in range(9):
            variables = []
            coefficients = [1 for _ in range(9)]
            for row in range(9):
                variables.append(getVarName(row, col, digit))
            rows.append([variables, coefficients])
            rhs.append(1)

    for digit in range(1, 10):
        for row in range(9):
            variables = []
            coefficients = [1 for _ in range(9)]
            for col in range(9):
                variables.append(getVarName(row, col, digit))
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
                        variables.append(getVarName(row, col, digit))
                rows.append([variables, coefficients])
                rhs.append(1)

    for row in range(9):
        for col in range(9):
            variables = []
            coefficients = [1 for _ in range(9)]
            for digit in range(1, 10):
                variables.append(getVarName(row, col, digit))
            rows.append([variables, coefficients])
            rhs.append(1)

    mySense = "E" * len(rows)
    myRowNames = ["r" + str(num) for num in range(len(rows))]
    prob.linear_constraints.add(lin_expr=rows, senses=mySense, rhs=rhs, names=myRowNames)
    return myColNames


def sudoku_solver(sudoku):
    myProb = cplex.Cplex()
    myColNames = lpInitialization(sudoku, myProb)
    myProb.parameters.mip.limits.populate.set(5)

    solution = []

    try:
        myProb.populate_solution_pool()
        for i in range(myProb.solution.pool.get_num()):
            x = myProb.solution.pool.get_values(i)
            sol = [[0 for i in range(9)] for i in range(9)]
            for row in range(9):
                for col in range(9):
                    for digit in range(1, 10):
                        pos = myColNames.index(getVarName(row, col, digit))
                        if x[pos] == 1:
                            if sudoku[row][col] > 0:
                                sol[row][col] = -1
                            else:
                                sol[row][col] = digit
            solution.append(sol)
        return solution
    except CplexError as exc:
        print(exc)
