import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os


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



os.makedirs("visualizations", exist_ok=True)

embedding_dims = [f'dim_{i}' for i in range(X.shape[1])]

correlations_pearson = pd.DataFrame(
    index=embedding_dims,
    columns=properties,
    dtype=float
)

correlations_spearman = pd.DataFrame(
    index=embedding_dims,
    columns=properties,
    dtype=float
)

for prop in properties:
    y_prop = combined[prop].dropna()
    valid_indices = y_prop.index

    for dim in embedding_dims:
        X_dim = combined.loc[valid_indices, dim]

        correlations_pearson.loc[dim, prop] = pearsonr(X_dim, y_prop)[0]
        correlations_spearman.loc[dim, prop] = spearmanr(X_dim, y_prop)[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

sns.heatmap(correlations_pearson.astype(float),
            cmap='RdBu_r', center=0, vmin=-0.3, vmax=0.3,
            cbar_kws={'label': 'Pearson Correlation'},
            ax=ax1)
ax1.set_title('Pearson Correlation: Embedding Dimensions × Glasgow Properties',
              fontsize=14, pad=20)
ax1.set_xlabel('Glasgow Properties', fontsize=12)
ax1.set_ylabel('Embedding Dimensions', fontsize=12)

sns.heatmap(correlations_spearman.astype(float),
            cmap='RdBu_r', center=0, vmin=-0.3, vmax=0.3,
            cbar_kws={'label': 'Spearman Correlation'},
            ax=ax2)
ax2.set_title('Spearman Correlation: Embedding Dimensions × Glasgow Properties',
              fontsize=14, pad=20)
ax2.set_xlabel('Glasgow Properties', fontsize=12)
ax2.set_ylabel('Embedding Dimensions', fontsize=12)

plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("Heatmap gespeichert: visualizations/correlation_heatmap.png")

print("\nTop 10 Korrelationen (Pearson):")
correlations_flat = []
for prop in properties:
    for dim in embedding_dims:
        correlations_flat.append({
            'Property': prop,
            'Dimension': dim,
            'Correlation': abs(correlations_pearson.loc[dim, prop])
        })

top_correlations = pd.DataFrame(correlations_flat).nlargest(10, 'Correlation')
print(top_correlations.to_string(index=False))

correlations_pearson.to_csv('visualizations/pearson_correlations.csv')
correlations_spearman.to_csv('visualizations/spearman_correlations.csv')

