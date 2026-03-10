# 2026-03-10

def vector_add(v1: list[float], v2: list[float]) -> list[float]:
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")
    return [a + b for a, b in zip(v1, v2)]

def dot_product(v1: list[float], v2: list[float]) -> float:
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")
    return sum(a * b for a, b in zip(v1, v2))

def matrix_vector(matrix: list[list[float]], vector: list[float]) -> list[float]:
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("Matrix columns must match vector size")
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]

def linear_regression(x: list[float], y: list[float]) -> tuple[float, float]:
    if len(x) != len(y):
        raise ValueError("Vectors must have the same length")
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    num = sum((xi - mean_x)*(yi - mean_y) for xi, yi in zip(x, y))
    den = sum((xi - mean_x)**2 for xi in x)
    if den == 0:
        raise ValueError("Denominator cannot be zero")
    a = num / den
    b = mean_y - (a * mean_x)
    return(a, b)

def avg_quadratic_error(x: list[float], y: list[float], a: float, b: float) -> float:
    if len(x) != len(y):
        raise ValueError("Vectors must have the same length")
    return sum((yi - (a * xi + b))**2 for xi, yi in zip(x, y)) / len(x)

# ------------------------------------------------------------------------------------


# addition de vecteurs
print(f"Addition de vecteurs\n{vector_add([1, 2, 3], [4, 5, 6])}\n")

# produit scalaire de vecteurs
print(f"Produit scalaire de vecteurs\n{dot_product([1, 2, 3], [4, 5, 6])}\n")

# produit matrice-vecteur
matrice = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
x = [2, 4, 6]
print(f"Produit matrice-vecteur\n{matrix_vector(matrice, x)}\n")

# régression linéaire
a, b = linear_regression([50, 70, 90], [100, 140, 180])
print(f"Régression linéaire\ny = {round(a, 2)}x + {round(b, 2)}\n")

# estimation d'erreur
datasets = [
    [
        [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0],
        [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    ],
    [
        [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0],
        [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 7.26, 7.26, 4.74]
    ],
    [
        [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0],
        [7.46, 6.77, 12.74, 7.11, 8.81, 8.84, 6.08, 5.39, 8.15, 6.40, 5.73]
    ],
    [
        [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 19.0, 8.0, 8.0, 8.0],
        [6.58, 5.76, 5.76, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]
    ]
]
print("MSE")
for set in datasets:
    v1 = set[0]
    v2 = set[1]
    a, b = linear_regression(v1, v2)
    print(f"y = {round(a, 3)}x + {round(b, 3)}, MSE = {round(avg_quadratic_error(v1, v2, a, b), 3)}")