from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
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


def train_model():
    properties = ['AROU_M', 'VAL_M', 'DOM_M', 'CNC_M', 'IMAG_M',
                  'FAM_M', 'AOA_M', 'SIZE_M', 'GEND_M']

    prop = properties[0]

    y = annotations[prop].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(
        f'Property {prop}',
        f'R² {r2_score(y_test, y_pred)}',
        f'RMSE {np.sqrt(mean_squared_error(y_test, y_pred))}',
        f'MAE {mean_absolute_error(y_test, y_pred)}', sep='\n'
    )

print(combined)
