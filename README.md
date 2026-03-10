## Sommaire

[day1.py](/2026-03-10/day1.py) :
- Somme de vecteurs
- Produit scalaire
- Produit matrice-vecteur
- Régression linéaire
- MSE

## Analyse

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

```
à noter qu'on empêche également l'insertion de valeurs incompatibles (vecteurs de longeurs différentes ou dénominateur égal à 0)
```

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
Voici une représentation graphique des datasets réalisée avec [matplotlib](https://matplotlib.org/) (code consultable dans [graph.py](/2026-03-10/graph.py)):

On remarquera que le dataset 3 possède le MSE le plus proche de 0, et les points sont effectivement les plus proche de notre fonction affine de prédiléction. Les datasets comme 1 ou 2 on une valeur d'intercept perceptiblement supérieure à 0, et le dataset 4 possède trop peu de variété de points pour établir une prédiléction ou corrélation pertinente, comme on peut l'observer.

![dataset_1](/plots/dataset_1.png)
![dataset_2](/plots/dataset_2.png)
![dataset_3](/plots/dataset_3.png)
![dataset_4](/plots/dataset_4.png)
