import pandas as pd

# Dataset from here https://files.grouplens.org/datasets/movielens/

df = pd.read_csv(
    "raw_data/ratings.dat",
    sep="::",
    engine="python",
    names=["user_id", "item_id", "rating", "timestamp"],
)

df["user_id"], _ = pd.factorize(df["user_id"])
df["item_id"], _ = pd.factorize(df["item_id"])
df["user_id"] += 1
df["item_id"] += 1

df = df.reset_index()
df_sorted = df.sort_values(by=["timestamp", "index"])
df_sorted = df_sorted.drop(columns=["index"])
df_sorted = df_sorted.reset_index(drop=True)

df_sorted.to_csv("data/ml1m.csv", index=False)
