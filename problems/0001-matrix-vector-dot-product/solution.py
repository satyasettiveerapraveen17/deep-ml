def matrix_dot_vector(a: list[list[int | float]], b: list[int | float]) -> list[int | float]:

    result = []

    if len(a[0]) != len(b):
        return -1

    for i in range(len(a)):
        total = 0

        for j in range(len(a[0])):
            total += a[i][j] * b[j]

        result.append(total)

    return result