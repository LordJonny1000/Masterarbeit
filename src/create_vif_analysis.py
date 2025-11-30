import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm
from pathlib import Path
import os

path = Path("/home/jonny/Schreibtisch/Multiling/Masterarbeit")
os.chdir(path)

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



print("=== MULTIKOLLINEARITÄTSANALYSE MIT VIF ===\n")

print("SCHRITT 1: VIF für Embedding-Dimensionen berechnen")
print("-" * 60)
print("Für jede Embedding-Dimension berechnen wir, wie gut sie durch")
print("alle anderen Dimensionen vorhergesagt werden kann.\n")

embedding_data = combined[embedding_dims].values

vif_results_embeddings = []

print(f"Berechne VIF für {len(embedding_dims)} Dimensionen...")
for i in tqdm(range(len(embedding_dims))):
    vif = variance_inflation_factor(embedding_data, i)
    vif_results_embeddings.append({
        'Dimension': embedding_dims[i],
        'VIF': vif
    })

vif_embeddings_df = pd.DataFrame(vif_results_embeddings)

print(f"\nStatistik der VIF-Werte (Embedding-Dimensionen):")
print(f"  Mittelwert: {vif_embeddings_df['VIF'].mean():.2f}")
print(f"  Median:     {vif_embeddings_df['VIF'].median():.2f}")
print(f"  Min:        {vif_embeddings_df['VIF'].min():.2f}")
print(f"  Max:        {vif_embeddings_df['VIF'].max():.2f}")
print(f"  Anzahl VIF > 10:  {(vif_embeddings_df['VIF'] > 10).sum()}")
print(f"  Anzahl VIF > 5:   {(vif_embeddings_df['VIF'] > 5).sum()}")

vif_embeddings_df.to_csv('visualizations/vif_embeddings.csv', index=False)


print("\n" + "="*60)
print("SCHRITT 2: VIF für Glasgow Properties berechnen")
print("-" * 60)
print("Für jede Glasgow Property berechnen wir, wie gut sie durch")
print("alle anderen Properties vorhergesagt werden kann.\n")

properties_data = combined[properties].dropna()

vif_results_properties = []

for i, prop in enumerate(properties):
    vif = variance_inflation_factor(properties_data.values, i)
    vif_results_properties.append({
        'Property': prop.replace('_M', ''),
        'VIF': vif
    })
    print(f"  {prop.replace('_M', ''):6s}: VIF = {vif:6.2f}")

vif_properties_df = pd.DataFrame(vif_results_properties)
vif_properties_df.to_csv('visualizations/vif_properties.csv', index=False)


print("\n" + "="*60)
print("SCHRITT 3: Visualisierung erstellen")
print("-" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].hist(vif_embeddings_df['VIF'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(5, color='orange', linestyle='--', linewidth=2, label='VIF = 5 (Schwelle)')
axes[0, 0].axvline(10, color='red', linestyle='--', linewidth=2, label='VIF = 10 (Schwelle)')
axes[0, 0].set_xlabel('VIF Value', fontsize=12)
axes[0, 0].set_ylabel('Number Dimensions', fontsize=12)
axes[0, 0].set_title('Distribution of VIF-Values (Embedding-Dimensions)', fontsize=13)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

sorted_vif = vif_embeddings_df.nlargest(30, 'VIF')
axes[0, 1].barh(range(len(sorted_vif)), sorted_vif['VIF'].values)
axes[0, 1].axvline(5, color='orange', linestyle='--', linewidth=2)
axes[0, 1].axvline(10, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('VIF Value', fontsize=12)
axes[0, 1].set_ylabel('Dimension', fontsize=12)
axes[0, 1].set_title('Top 30 Dimensions with highest VIF', fontsize=13)
axes[0, 1].set_yticks(range(len(sorted_vif)))
axes[0, 1].set_yticklabels(sorted_vif['Dimension'].values, fontsize=8)
axes[0, 1].invert_yaxis()
axes[0, 1].grid(True, alpha=0.3, axis='x')

axes[1, 0].bar(range(len(vif_properties_df)), vif_properties_df['VIF'].values,
               color='steelblue', edgecolor='black', alpha=0.7)
axes[1, 0].axhline(5, color='orange', linestyle='--', linewidth=2, label='VIF = 5')
axes[1, 0].axhline(10, color='red', linestyle='--', linewidth=2, label='VIF = 10')
axes[1, 0].set_xlabel('Glasgow Property', fontsize=12)
axes[1, 0].set_ylabel('VIF Value', fontsize=12)
axes[1, 0].set_title('VIF-Value for Glasgow Properties', fontsize=13)
axes[1, 0].set_xticks(range(len(vif_properties_df)))
axes[1, 0].set_xticklabels(vif_properties_df['Property'].values, rotation=45, ha='right')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

dimension_indices = [int(d.split('_')[1]) for d in vif_embeddings_df['Dimension']]
axes[1, 1].scatter(dimension_indices, vif_embeddings_df['VIF'].values,
                   alpha=0.5, s=20, color='steelblue')
axes[1, 1].axhline(5, color='orange', linestyle='--', linewidth=2, label='VIF = 5')
axes[1, 1].axhline(10, color='red', linestyle='--', linewidth=2, label='VIF = 10')
axes[1, 1].set_xlabel('Dimension Index', fontsize=12)
axes[1, 1].set_ylabel('VIF Value', fontsize=12)
axes[1, 1].set_title('VIF-Value across all Dimensions', fontsize=13)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/vif_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualisierung gespeichert: visualizations/vif_analysis.png")

print("\n" + "="*60)
print("INTERPRETATION:")
print("-" * 60)
print("Embedding-Dimensionen:")
if vif_embeddings_df['VIF'].median() < 5:
    print("  ✓ Median VIF < 5: Geringe Multikollinearität zwischen Dimensionen")
elif vif_embeddings_df['VIF'].median() < 10:
    print("  ⚠ Median VIF 5-10: Moderate Multikollinearität")
else:
    print("  ✗ Median VIF > 10: Starke Multikollinearität (problematisch)")

print("\nGlasgow Properties:")
max_vif_prop = vif_properties_df.loc[vif_properties_df['VIF'].idxmax()]
print(f"  Höchster VIF: {max_vif_prop['Property']} = {max_vif_prop['VIF']:.2f}")
if max_vif_prop['VIF'] > 10:
    print(f"  ⚠ {max_vif_prop['Property']} zeigt starke Multikollinearität!")