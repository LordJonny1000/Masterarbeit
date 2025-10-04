import pandas as pd

df = pd.read_csv("data/glasgow_norms.csv", header=[0, 1])

# Nur die "M" (Mean) Unterspalten behalten + "Words" & "Length"
df = df.loc[:, df.columns.get_level_values(1).isin(["M"]) | df.columns.get_level_values(0).isin(["Words", "Length"])]

# Spaltennamen vereinfachen
df.columns = [
    f"{top}_M" if sub == "M" else top
    for top, sub in df.columns
]

glasgow = df

with open("data/numberbatch_english.txt", "r") as f:
     numberbatch = {line.split()[0].removeprefix('/c/en/'): line.split()[1:] for line in f.readlines()}


print(numberbatch)