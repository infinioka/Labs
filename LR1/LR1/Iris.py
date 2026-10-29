# Датасет 1 (Iris): EDA + регрессия; графики -> pics/
from pathlib import Path
FIG_DIR = Path(__file__).resolve().parent / 'pics'
FIG_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pingouin as pg
from scipy import stats

iris = load_iris()
iris_pd = pd.DataFrame(data=iris.data, columns=iris.feature_names)


print(iris_pd.head())
print(iris_pd.info())
print(iris_pd.describe())
print(iris_pd.shape)


plt.figure(figsize=(11, 6))

plt.boxplot(iris_pd.to_numpy(),tick_labels=iris_pd.columns)

plt.title("Диаграмма Тьюки для Iris")
plt.ylabel("Размер, см")
plt.show()
plt.savefig(FIG_DIR / "1_1_Диаграмма_Тьюки.png")

import seaborn as sns 
corr_matrix = iris_pd.corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot = True,fmt=".2f",cmap="coolwarm",  vmin=-1, vmax=1)
plt.title("Тепловая карта парных коэффициентов корреляции")
plt.show()
plt.savefig(FIG_DIR / "1_2_Тепловая_карта_парных_корреляций.png")
print(corr_matrix)

partial_corr_matrix = iris_pd.pcorr()
plt.figure(figsize=(9, 7))
sns.heatmap(partial_corr_matrix, annot = True,fmt=".2f",cmap="coolwarm",  vmin=-1, vmax=1, center = 0)
plt.title("Тепловая карта частных коэффициентов корреляции")
plt.show()
plt.savefig(FIG_DIR / "1_3_Тепловая_карта_частных_корреляций.png")
print(partial_corr_matrix)

difference = partial_corr_matrix - corr_matrix

print("Разница между частными и парными корреляциями:")
print(difference.round(3))

print("Тест Колмогорова-Смирнова. Уровень значимости α = 0.05")

for col in iris_pd.columns:
    x = iris_pd[col]

    x_mean = np.mean(x)
    std = np.std(x, ddof=1)

    x_std = (x - x_mean) / std

    statistic, pvalue = stats.kstest(
        x_std, "norm", alternative="two-sided"
    )

    if pvalue > 0.05:
        result = "H0 не отвергается"
    else:
        result = "H0 отвергается"

    print(f"\nПризнак: {col}")
    print(f"Статистика D: {statistic:.4f}")
    print(f"p-value: {pvalue:.4f}")
    print(f"Вывод: {result}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, col in zip(axes.flatten(), iris_pd.columns):
    ax.hist(
        iris_pd[col],
        bins=12,
        edgecolor="black"
    )

    ax.set_title(col)
    ax.set_xlabel("Значение признака, см")
    ax.set_ylabel("Частота")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
plt.savefig(FIG_DIR / "1_4_Гистограммы_признаков.png")

variation = pd.DataFrame({
    "Среднее": iris_pd.mean(),
    "СКО": iris_pd.std(),
    "Коэффициент вариации (%)":
        iris_pd.std() / iris_pd.mean().abs() * 100
})

print(variation.round(3))


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


x1 = iris_pd["sepal length (cm)"]
x2 = iris_pd["sepal width (cm)"]
x3 = iris_pd["petal length (cm)"]
y = iris_pd["petal width (cm)"]


X_pair = iris_pd[["petal length (cm)"]]


X_multiple = iris_pd[["sepal length (cm)", "sepal width (cm)","petal length (cm)"]]

print("Размер парной выборки:", X_pair.shape)
print("Размер множественной выборки:", X_multiple.shape)
print("Размер выходной переменной:", y.shape)

# Создаём модель парной регрессии
pair_model = LinearRegression()

# Обучаем модель
pair_model.fit(X_pair, y)

# Получаем предсказанные значения
y_pred_pair = pair_model.predict(X_pair)

# Коэффициенты уравнения
b0_pair = pair_model.intercept_
b1_pair = pair_model.coef_[0]

print("Уравнение парной регрессии:")
print(f"Y = {b0_pair:.4f} + ({b1_pair:.4f}) * X3")

plt.figure(figsize=(10, 6))

plt.scatter(x3, y, label="Исходные значения")

# Сортируем X для правильного отображения линии
order = np.argsort(x3.to_numpy())

plt.plot(
    x3.to_numpy()[order],
    y_pred_pair[order],
    color="red",
    label="Парная регрессия"
)

plt.title("Парная регрессия: зависимость X4 от X3")
plt.xlabel("Длина лепестка X3, см")
plt.ylabel("Ширина лепестка X4, см")
plt.legend()
plt.grid(alpha=0.3)

plt.show()
plt.savefig(FIG_DIR / "1_5_Парная_регрессия.png")

# Создаём модель множественной регрессии
multiple_model = LinearRegression()

# Обучение
multiple_model.fit(X_multiple, y)

# Предсказание
y_pred_multiple = multiple_model.predict(X_multiple)

# Коэффициенты
b0 = multiple_model.intercept_
b1, b2, b3 = multiple_model.coef_

print("Уравнение множественной регрессии:")
print(
    f"Y = {b0:.4f} "
    f"+ ({b1:.4f}) * X1 "
    f"+ ({b2:.4f}) * X2 "
    f"+ ({b3:.4f}) * X3"
)

plt.figure(figsize=(12, 6))

# Номера наблюдений
n = np.arange(len(y))

# Реальные значения
plt.plot(
    n, y,
    color="black",
    label="Исходные значения",
    alpha=0.6
)

# Предсказания парной регрессии
plt.plot(
    n, y_pred_pair,
    color="red",
    label="Парная регрессия",
    alpha=0.8
)

# Предсказания множественной регрессии
plt.plot(
    n, y_pred_multiple,
    color="green",
    label="Множественная регрессия",
    alpha=0.8
)

plt.title("Сравнение предсказаний моделей")
plt.xlabel("Номер наблюдения")
plt.ylabel("Ширина лепестка, см")
plt.legend()
plt.grid(alpha=0.3)

plt.show()
plt.savefig(FIG_DIR / "1_6_Сравнение_предсказаний.png")

# Остатки парной регрессии
residuals_pair = y.to_numpy() - y_pred_pair

# Остатки множественной регрессии
residuals_multiple = y.to_numpy() - y_pred_multiple

# Один рисунок с двумя графиками
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Парная регрессия
axes[0].scatter(
    y_pred_pair,
    residuals_pair,
    color="red"
)
axes[0].axhline(0, color="black", linestyle="--")
axes[0].set_title("Парная регрессия")
axes[0].set_xlabel("Предсказанные значения")
axes[0].set_ylabel("Остатки")
axes[0].grid(alpha=0.3)

# Множественная регрессия
axes[1].scatter(
    y_pred_multiple,
    residuals_multiple,
    color="green"
)
axes[1].axhline(0, color="black", linestyle="--")
axes[1].set_title("Множественная регрессия")
axes[1].set_xlabel("Предсказанные значения")
axes[1].set_ylabel("Остатки")
axes[1].grid(alpha=0.3)

plt.suptitle("Зависимость остатков от предсказанных значений")
plt.tight_layout()
plt.show()
plt.savefig(FIG_DIR / "1_7_Остатки_от_предсказанных.png")

def regression_metrics(y_true, y_pred, p):
    n = len(y_true)

    # Коэффициент детерминации
    r2 = r2_score(y_true, y_pred)

    # Скорректированный R²
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # Сумма квадратов остатков
    sse = np.sum((np.asarray(y_true) - y_pred) ** 2)

    # Стандартная ошибка регрессии
    standard_error = np.sqrt(sse / (n - p - 1))

    # Множественный коэффициент корреляции
    r = np.sqrt(r2)

    return {
        "R": r,
        "R²": r2,
        "Скорректированный R²": r2_adj,
        "Стандартная ошибка": standard_error
    }


# Парная регрессия: один входной признак
metrics_pair = regression_metrics(y, y_pred_pair, p=1)

# Множественная регрессия: три входных признака
metrics_multiple = regression_metrics(y, y_pred_multiple, p=3)

comparison = pd.DataFrame({
    "Парная регрессия": metrics_pair,
    "Множественная регрессия": metrics_multiple
})

print(comparison.round(4))

# Проверяем основные характеристики остатков

residuals_df = pd.DataFrame({
    "Парная регрессия": residuals_pair,
    "Множественная регрессия": residuals_multiple
})

print(residuals_df.describe())

from scipy import stats

print("Критерий Колмогорова-Смирнова")
print("Уровень значимости α = 0.05\n")

for name, residuals in residuals_df.items():

    # Среднее и выборочное СКО
    mean = np.mean(residuals)
    std = np.std(residuals, ddof=1)

    # Стандартизация остатков
    residuals_std = (residuals - mean) / std

    # Критерий Колмогорова-Смирнова
    statistic, pvalue = stats.kstest(
        residuals_std,
        "norm",
        alternative="two-sided"
    )

    if pvalue > 0.05:
        result = "H0 не отвергается"
    else:
        result = "H0 отвергается"

    print(name)
    print(f"Статистика D = {statistic:.4f}")
    print(f"p-value = {pvalue:.4f}")
    print(f"Вывод: {result}\n")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(
    residuals_pair,
    bins=15,
    edgecolor="black"
)
axes[0].set_title("Остатки парной регрессии")
axes[0].set_xlabel("Остатки")
axes[0].set_ylabel("Частота")

axes[1].hist(
    residuals_multiple,
    bins=15,
    edgecolor="black"
)
axes[1].set_title("Остатки множественной регрессии")
axes[1].set_xlabel("Остатки")
axes[1].set_ylabel("Частота")

plt.tight_layout()
plt.show()
plt.savefig(FIG_DIR / "1_8_Гистограммы_остатков.png")

def calculate_dw(residuals):
    residuals = np.asarray(residuals)
    numerator = np.sum(np.diff(residuals) ** 2)
    denominator = np.sum(residuals ** 2)
    return numerator / denominator

print("Парная:", calculate_dw(residuals_pair))
print("Множественная:", calculate_dw(residuals_multiple))

alpha = 0.05

results = []

models = {
    "Парная регрессия": residuals_pair,
    "Множественная регрессия": residuals_multiple
}

for name, residuals in models.items():

    # Стандартизация остатков
    mean = np.mean(residuals)
    std = np.std(residuals, ddof=1)
    residuals_std = (residuals - mean) / std

    # Критерий Колмогорова-Смирнова
    statistic, pvalue = stats.kstest(
        residuals_std,
        "norm",
        alternative="two-sided"
    )

    # Статистика Дурбина-Уотсона
    dw = calculate_dw(residuals)

    # Проверка гипотезы о нормальности
    if pvalue > alpha:
        normality = "H0 не отвергается"
    else:
        normality = "H0 отвергается"

    # Предварительная интерпретация DW
    if dw < 2:
        dw_result = "Возможна положительная автокорреляция"
    elif dw > 2:
        dw_result = "Возможна отрицательная автокорреляция"
    else:
        dw_result = "DW = 2"

    results.append({
        "Модель": name,
        "Статистика D": statistic,
        "p-value": pvalue,
        "Вывод по H0": normality,
        "DW": dw,
        "Вывод по DW": dw_result
    })

results_df = pd.DataFrame(results)

print(results_df.round(4))

plt.figure(figsize=(11, 6))

plt.boxplot(
    iris_pd.to_numpy(),
    tick_labels=iris_pd.columns
)

plt.title("Анализ выбросов в датасете Iris")
plt.xlabel("Признаки")
plt.ylabel("Размер, см")
plt.grid(axis="y", alpha=0.3)

plt.show()
plt.savefig(FIG_DIR / "1_9_Анализ_выбросов.png")

outliers_results = []

for col in iris_pd.columns:

    # Первый и третий квартили
    q1 = iris_pd[col].quantile(0.25)
    q3 = iris_pd[col].quantile(0.75)

    # Межквартильный размах
    iqr = q3 - q1

    # Границы выбросов
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    # Определение выбросов
    outliers = iris_pd[
        (iris_pd[col] < lower) |
        (iris_pd[col] > upper)
    ]

    outliers_results.append({
        "Признак": col,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Нижняя граница": lower,
        "Верхняя граница": upper,
        "Количество выбросов": len(outliers)
    })

outliers_df = pd.DataFrame(outliers_results)

print(outliers_df.round(3))

for col in iris_pd.columns:

    q1 = iris_pd[col].quantile(0.25)
    q3 = iris_pd[col].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = iris_pd.loc[
        (iris_pd[col] < lower) |
        (iris_pd[col] > upper),
        col
    ]

    print(f"\nПризнак: {col}")
    print("Значения выбросов:", outliers.to_list())

from sklearn.model_selection import train_test_split

# Входные переменные
X = iris_pd[
    ["sepal length (cm)",
     "sepal width (cm)",
     "petal length (cm)"]
]

# Выходная переменная
y = iris_pd["petal width (cm)"]

# Разделение выборки
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("Размер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)



# Создание модели
model_test = LinearRegression()

# Обучение на 75% выборки
model_test.fit(X_train, y_train)

# Предсказания на тестовой выборке
y_pred_test = model_test.predict(X_test)

print("Коэффициенты модели:")
print("b0 =", model_test.intercept_)
print("b1, b2, b3 =", model_test.coef_)

from sklearn.metrics import mean_squared_error, r2_score

# Среднеквадратическая ошибка
mse = mean_squared_error(y_test, y_pred_test)

# Корень из среднеквадратической ошибки
rmse = np.sqrt(mse)

# Коэффициент детерминации
r2 = r2_score(y_test, y_pred_test)

print(f"MSE: {mse:.4f}")
print(f"СКО (RMSE): {rmse:.4f} см")
print(f"Коэффициент детерминации R²: {r2:.4f}")

# Ошибки на тестовой выборке
errors = y_test.to_numpy() - y_pred_test

plt.figure(figsize=(10, 6))

plt.scatter(
    y_pred_test,
    errors,
    color="blue",
    label="Ошибки прогнозирования"
)

# Горизонтальная линия нулевой ошибки
plt.axhline(
    y=0,
    color="red",
    linestyle="--",
    label="Нулевая ошибка"
)

plt.title("График ошибок множественной регрессии")
plt.xlabel("Предсказанная ширина лепестка, см")
plt.ylabel("Ошибка прогнозирования, см")

plt.legend()
plt.grid(alpha=0.3)
plt.show()
plt.savefig(FIG_DIR / "1_10_График_ошибок.png")

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred_test,
    color="green",
    label="Предсказания"
)

# Линия идеального совпадения
min_value = min(y_test.min(), y_pred_test.min())
max_value = max(y_test.max(), y_pred_test.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    color="red",
    linestyle="--",
    label="Идеальное предсказание"
)

plt.title("Реальные и предсказанные значения")
plt.xlabel("Реальная ширина лепестка, см")
plt.ylabel("Предсказанная ширина лепестка, см")

plt.legend()
plt.grid(alpha=0.3)
plt.show()
plt.savefig(FIG_DIR / "1_11_Реальные_и_предсказанные.png")
