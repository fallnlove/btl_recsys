from pathlib import Path

import pandas as pd
from tqdm import tqdm

optuna_folder = Path("/Users/arturgimranov/Downloads/optuna_outputs2")
hp_names = [
    "embedding_dim",
    "num_layers",
    "num_hops",
    "num_heads",
    "dim_feedforward",
    "dropout",
    "lr",
]
col_names = [
    "model",
    "dataset",
    "train_q",
    "val_q",
    "test/ndcg@10",
    "val/ndcg@10",
    "num_iter",
]
results = pd.DataFrame(columns=(col_names + hp_names))
for dir in optuna_folder.iterdir():
    if not dir.name.startswith("."):
        # print(dir)
        if (dir / "index.csv").exists():
            result_df = pd.read_csv(dir / "index.csv")
            model, dataset, train_q, val_q = dir.name.split("_")[:4]
            argmax_row = result_df.iloc[result_df["val/ndcg@10"].argmax()].to_dict()
            col_values = [
                model,
                dataset,
                train_q,
                val_q,
                argmax_row["test/ndcg@10"],
                argmax_row["val/ndcg@10"],
                len(result_df),
            ]
            hp_values = [argmax_row[hp_name] for hp_name in hp_names]
            results.loc[len(results)] = col_values + hp_values

print(len(results))
results.to_csv("all_results.csv", index=False)
print()
print("SASREC")
sasrec_df = results[results.model == "sasrec"].sort_values(by="dataset")
print(sasrec_df[sasrec_df.train_q == "098"])
sasrec_df[sasrec_df.train_q == "098"].to_csv("sasrec_098_096.csv", index=False)
print()
print("MRGSREC")
mrgsrec_df = results[results.model == "mrgsrec"].sort_values(by="dataset")
print(mrgsrec_df[mrgsrec_df.train_q == "098"])
mrgsrec_df[mrgsrec_df.train_q == "098"].to_csv("mrgsrec_098_096.csv", index=False)
print()

print()
print("SASREC")
sasrec_df = results[results.model == "sasrec"].sort_values(by="dataset")
print(sasrec_df[sasrec_df.train_q == "08"])
sasrec_df[sasrec_df.train_q == "08"].to_csv("sasrec_08_09.csv", index=False)
print()
print("MRGSREC")
mrgsrec_df = results[results.model == "mrgsrec"].sort_values(by="dataset")
print(mrgsrec_df[mrgsrec_df.train_q == "08"])
mrgsrec_df[mrgsrec_df.train_q == "08"].to_csv("mrgsrec_08_09.csv", index=False)
print()

# df_08_09 = pd.read_csv(
#     "/Users/arturgimranov/Downloads/optuna_outputs/mrgsrec_ml1m_08_09_fixed_10/index.csv"
# )
# print(df_08_09.iloc[df_08_09["val/ndcg@10"].argmax()])

# df_098_096 = pd.read_csv(
#     "/Users/arturgimranov/Downloads/optuna_outputs/mrgsrec_ml1m_098_096/index.csv"
# )
# print(df_098_096.iloc[df_098_096["val/ndcg@10"].argmax()])
