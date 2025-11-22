from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np

df = pd.read_csv("data/glasgow_norms.csv", header=[0, 1])

df = df.loc[:, df.columns.get_level_values(1).isin(["M"]) |
            df.columns.get_level_values(0).isin(["Words", "Length"])]

df.columns = [f"{top}_M" if sub == "M" else top for top, sub in df.columns]

with open("data/numberbatch_english.txt", "r") as f:
    numberbatch = {line.split()[0].removeprefix('/c/en/'):
                   np.array([float(x) for x in line.split()[1:]], dtype=np.float32)
                   for line in f}

glasgow_dict = df.set_index('Words').to_dict('index')
words_in_both = set(glasgow_dict.keys()) & set(numberbatch.keys())

X = np.array([numberbatch[w] for w in words_in_both])
y = df[df['Words'].isin(words_in_both)].set_index('Words').loc[list(words_in_both)]
words = list(words_in_both)

data = pd.DataFrame(X, index=words, columns=[f'dim_{i}' for i in range(X.shape[1])])
annotations = y

combined = pd.concat([data, annotations], axis=1)

properties = ['AROU_M', 'VAL_M', 'DOM_M', 'CNC_M', 'IMAG_M',
              'FAM_M', 'AOA_M', 'SIZE_M', 'GEND_M']

with open("analysis/linear_inspection.txt", 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("LINEAR REGRESSION ANALYSIS - GLASGOW NORMS\n")
    f.write("=" * 80 + "\n\n")

for prop in properties:
    y = annotations[prop].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    weights = model.coef_
    bias = model.intercept_
    weights_dict = {num: float(weight) for num, weight in enumerate(weights)}
    weights_sorted = sorted(weights_dict.items(), key=lambda x: x[1])
    top_neg = weights_sorted[:20]
    top_pos = weights_sorted[-20:][::-1]

    with open("analysis/linear_inspection.txt", 'a', encoding='utf-8') as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"PROPERTY: {prop}\n")
        f.write(f"{'=' * 80}\n\n")

        f.write(f"Performance Metrics:\n")
        f.write(f"  Train R²: {train_r2:.4f}  |  Test R²: {test_r2:.4f}  |  Diff: {train_r2 - test_r2:.4f}\n")
        f.write(f"  Train RMSE: {train_rmse:.4f}  |  Test RMSE: {test_rmse:.4f}\n")
        f.write(f"  Train MAE: {train_mae:.4f}  |  Test MAE: {test_mae:.4f}\n")
        f.write(f"  Intercept: {bias:.4f}\n\n")

        f.write(f"Top 20 Positive Weights (most positive correlation):\n")
        for dim, weight in top_pos:
            f.write(f"  Dim {dim:3d}: {weight:8.4f}\n")

        f.write(f"\nTop 20 Negative Weights (most negative correlation):\n")
        for dim, weight in top_neg:
            f.write(f"  Dim {dim:3d}: {weight:8.4f}\n")

        f.write(f"\n")


#
# model = SVR(kernel='rbf')
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
#
# print(
#     f'Property {prop}',
#     f'R² {r2_score(y_test, y_pred)}',
#     f'RMSE {np.sqrt(mean_squared_error(y_test, y_pred))}',
#     f'MAE {mean_absolute_error(y_test, y_pred)}', sep='\n'
# )




def compute_average_correlation(dimension_indices, property_name):
    from scipy.stats import pearsonr

    if not dimension_indices:
        return 0.0

    correlations = []
    for dim_idx in dimension_indices:
        if dim_idx < 0 or dim_idx >= X.shape[1]:
            continue

        dim_col = f'dim_{dim_idx}'
        dim_values = combined[dim_col].values
        property_values = combined[property_name].values

        mask = ~(np.isnan(dim_values) | np.isnan(property_values))
        if mask.sum() < 2:
            continue

        corr, _ = pearsonr(dim_values[mask], property_values[mask])
        correlations.append(abs(corr))

    return np.mean(correlations) if correlations else 0.0

#

