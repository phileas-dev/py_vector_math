## Sommaire

- [Analyse Jour 1](#analyse-j1)
- [Analyse Jour 2](#analyse-j2)
- [TP Gradient Descent](#tp-gradient-descent)
- [TP Extremum](/2026-03-11/extremum.md)
- [L'ensemble des graphes](/plots/)

## Analyse (J1)

Une fonction de prédiction paramétrée prend des **données en entrée** et produit une **prédiction en sortie**, avec des paramètres ajustables. Son rôle consiste à **modéliser une relation** entre 2 vecteurs $X$ et $Y$, puis de généraliser pour établir des projections.

Pour rappel des formules:

### Régression linéaire
$$a = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}$$
$$b = \bar{y} - a\bar{x}$$
- $a$ = la pente (slope) ou la dérivée
- $b$ = l'ordonnée à l'origine (intercept)

### Fonction de coût (MSE)
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n}(Y_i - \hat{Y}_i)^2$$
- **MSE** = erreur quadratique moyenne
- $n$ = nombre de points de données
- $Y_i$ = valeurs observées
- $\hat{Y}_i$ = valeurs prévisionnelles

---

Les datasets ont été écrits en Python en tant que listes imbriquées:

```py
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
```

voici la fonction `linear_regression`:

```py
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
```

1. On calcule les **moyennes** de chaque vecteur, soit la somme des valeurs du vecteurs divisée par le nombre de valeurs (`sum(n) / len(n)`)

2. On calcule le **numérateur** et le **dénominateur** de la pente séparément, pour la lisibilité. Pour effectuer les sommes, on utilise une **generator expression** (équivalent à une boucle sur une seule ligne) qui itère sur le résultat de la fonction zip(x,y). Cette fonction permet de combiner nos listes en liste de tuples `[(x1,y1), (x2,y2), (x3,y3), ...]`

3. On réutilise les moyennes pour déterminer l'ordonnée b et on return un tuple contenant les deux valeurs a et b.
à noter qu'on empêche également l'insertion de valeurs incompatibles (vecteurs de longeurs différentes ou dénominateur égal à 0)


On applique:

```py
a, b = linear_regression(datasets[0][0], datasets[0][1])
print(f"Régression linéaire\ny = {round(a, 2)}x + {round(b, 2)}\n")
```
```
>>> Régression linéaire
>>> y = 0.5x + 3.0
```

Implémentons à présent la fonction de coût, qui a besoin des valueurs a et b calculées précédemment:

```py
def avg_quadratic_error(x: list[float], y: list[float], a: float, b: float) -> float:
    if len(x) != len(y):
        raise ValueError("Vectors must have the same length")
    return sum((yi - (a * xi + b))**2 for xi, yi in zip(x, y)) / len(x)
```
On applique:

```py
for set in datasets:
    v1 = set[0]
    v2 = set[1]
    a, b = linear_regression(v1, v2)
    print(f"y = {round(a, 3)}x + {round(b, 3)}, MSE = {round(avg_quadratic_error(v1, v2, a, b), 3)}")
```
```
>>> y = 0.5x + 3.0, MSE = 1.251
>>> y = 0.449x + 3.29, MSE = 1.471
>>> y = 0.518x + 2.925, MSE = 1.205
>>> y = 0.518x + 2.665, MSE = 1.309
```

Voici une représentation graphique des datasets réalisée avec [matplotlib](https://matplotlib.org/) (code consultable dans [graph.py](/2026-03-10/graph.py)):

On remarquera que le dataset 3 possède le MSE le plus proche de 0, et les points sont effectivement les plus proches de notre fonction affine de prédiléction. Les datasets comme 1 ou 2 on une valeur d'intercept perceptiblement supérieure à 0, et le dataset 4 possède trop peu de variété de points pour établir une prédiléction ou corrélation pertinente, comme on peut l'observer.

![dataset_1](/plots/dataset_1.png)
![dataset_2](/plots/dataset_2.png)
![dataset_3](/plots/dataset_3.png)
![dataset_4](/plots/dataset_4.png)



## Analyse (J2)

On reprend le dataset 3 qui est le plus pertinent. On va déterminer le modèle de prédiléction sous forme matricielle cette fois-ci, avec $y=Xθ$

$$\theta = \begin{bmatrix} b \\ a \end{bmatrix}$$

$$\theta = (X^T X)^{-1} X^T y$$

où :
- $X^T$ = transposée
- $(X^T X)^{-1}$ = inverse de matrice

On utilise numpy dans [day2.py](/2026-03-11/day2.py) comme ceci:

```py
X = np.array([10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0])
Y = np.array([7.46, 6.77, 12.74, 7.11, 8.81, 8.84, 6.08, 5.39, 8.15, 6.40, 5.73])

X_matrix = np.column_stack((np.ones(len(X)), X))
theta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ Y
b = theta[0]
a = theta[1]

print(f"pente a = {a}")
print(f"biais b = {b}")
```
```
>>> pente a = 0.5182727272727278
>>> biais b = 2.9246363636363584
```

On retrouve bien le même résultat qu'hier. La différence majeure ici est l'écriture de $θ$ qui représente une matrice de paramètres.

![dataset_3](/plots/dataset_3.png)

---

## TP Gradient Descent

Le script dans [day2.py](/2026-03-11/day2.py) pour calculer le gradient descent est le suivant:

```py
lr = 0.01
iterations = 200
w = 0
b = min(zip(X, Y))[1]

loss_history = []
w_history = []
b_history = []

for i in range(iterations):
    y_pred = w * X + b

    loss = np.mean((Y - y_pred) ** 2)
    loss_history.append(loss)

    dw = (-2 / X.size) * np.sum(X * (Y - y_pred))
    db = (-2 / X.size) * np.sum(Y - y_pred)

    w = w - lr * dw
    b = b - lr * db

    w_history.append(w)
    b_history.append(b)

print("Gradient Descent")
print("w =", w)
print("b =", b)
```

Les variables `lr`, `w`, `b`, et `iterations` ont été ajustées pour aligner les résultats le plus possible. On a notamment baissé le learning rate pour une courbe de loss plus précise, et changé l'ordonnée b pour correspondre au dataset.

Voici la courbe de loss:

![gradient descent loss curve](/plots/loss_curve.png)

Elle est descendante donc elle converge.

Le résultat approximé par la descente de gradient est représenté par la droite ci-dessous:

![gradient descent](/plots/gradient_descent.png)

Le script complet compare les 3 méthodes avec ce résultat:

```
Numpy régression linéaire
pente a = 0.5182727272727278
biais b = 2.9246363636363584

Gradient Descent
w = 0.3586633264524029
b = 4.5365348170736715

Scikit-learn
coef = 0.5182727272727274
intercept = 2.9246363636363633
```

L'implémentation scikit-learn est robuste mais aussi précise que notre régression linéaire précédente. En revanche, l'approche par descente de gradient est très sensible aux conditions initiales: un learning rate (="pas") trop élevé ou un nombre d'itérations trop extrême rend les résultats non-interprétables.