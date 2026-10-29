# Корреляционный анализ Barotrop/Chemical/Factor; графики -> pics/
from pathlib import Path
FIG_DIR = Path(__file__).resolve().parent / 'pics'
FIG_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pathlib import Path as _P
_HERE = _P(__file__).resolve().parent if "__file__" in globals() else _P.cwd()
DATA_DIR = _HERE / "datasets" / "csv"
if not DATA_DIR.is_dir():
    DATA_DIR = _HERE.parent / "datasets" / "csv"


def correlation_analysis(df, name):
    if name == "Barotrop.csv":
         df = df.drop(columns=["CLASS"])

    # Выбираем только количественные признаки
    numeric_df = df.select_dtypes(include="number")

    # Удаляем столбцы, где меньше двух различных значений
    numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]

    if numeric_df.shape[1] < 2:
        print(f"{name}: недостаточно числовых признаков")
        return None

    # Парные коэффициенты корреляции Пирсона
    corr_matrix = numeric_df.corr(method="pearson")

    print(f"\nДатасет: {name}")
    print("Размер:", df.shape)

    print(corr_matrix.round(3))

    # Тепловая карта
    plt.figure(figsize=(9, 7))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0
    )

    plt.title(f"Тепловая карта корреляций — {name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'corr_{Path(name).stem.lower()}.png')
    plt.show()

    # Получаем уникальные пары признаков
    pairs = []

    columns = corr_matrix.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            r = corr_matrix.iloc[i, j]

            if pd.notna(r):
                pairs.append({
                    "Признак X": columns[i],
                    "Признак Y": columns[j],
                    "Корреляция r": r,
                    "|r|": abs(r)
                })

    pairs_df = pd.DataFrame(pairs)

    pairs_df = pairs_df.sort_values(
        by="|r|",
        ascending=False
    )

    print("Пары признаков по силе корреляции:")
    print(pairs_df.round(3))

    return corr_matrix, pairs_df

files = [
    "Barotrop.csv",
    "Chemical_process.csv",
    "Factor.csv"
]

results = {}

for filename in files:
    df = pd.read_csv(DATA_DIR / filename)

    print(f"\nДатасет: {filename}")
    print(df.head())

    results[filename] = correlation_analysis(
        df, filename
    )
