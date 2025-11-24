import json
import pprint
from pathlib import Path

datasets_mapping = {
    "reviews_Beauty_5": "beauty",
    "reviews_Clothing_Shoes_and_Jewelry_5": "clothing",
    "reviews_Sports_and_Outdoors_5": "sports",
    "ml1m": "ml1m",
}

root = Path("/Users/arturgimranov/Downloads/2025-11-12_results")

metrics_dict = {}

for subdir in root.iterdir():
    if not subdir.is_dir():
        continue

    parts = subdir.name.split("__")
    if len(parts) < 2:
        continue

    model_name = parts[0]
    dataset_name = parts[1].split("_098_096")[0]

    metrics_path = subdir / "metrics.json"
    if not metrics_path.exists():
        continue

    with open(metrics_path) as f:
        metrics = json.load(f)

    metrics_dict.setdefault(model_name, {})[datasets_mapping[dataset_name]] = metrics


# pprint.pprint(metrics_dict)

metric_names = [
    "val/ndcg@10",
    "val/coverage@10",
    "val/recall@10",
    "test/ndcg@10",
    "test/coverage@10",
    "test/recall@10",
]


def make_block(dataset):
    ans = []
    for metric_name in metric_names:
        sasrec_metric = metrics_dict["sasrec"].get(dataset, {}).get(metric_name, "tbd")
        mrgsrec_metric = (
            metrics_dict["mrgsrec"].get(dataset, {}).get(metric_name, "tbd")
        )
        if isinstance(sasrec_metric, float):
            sasrec_metric = f"{sasrec_metric:.4f}"
        if isinstance(mrgsrec_metric, float):
            mrgsrec_metric = f"{mrgsrec_metric:.4f}"
        ans.append(f"& {metric_name} & {sasrec_metric} & {mrgsrec_metric} \\\\")
    return "\n".join(ans)


# print(make_block("ml1m"))

s = f"""
\\begin{{table*}}[ht]
\\centering
\\caption{{Experimental results on benchmark datasets}}
\\begin{{tabular}}{{cl|cc}}
\\hline
Dataset & Metric & SASRec & MRGSRec \\\\
\\hline
\\multirow{{4}}{{*}} {{Beauty}}
{make_block("beauty")}
\\hline
\\multirow{{4}}{{*}} {{Clothing}}
{make_block("clothing")}
\\hline
\\multirow{{4}}{{*}} {{Sports}}
{make_block("sports")}
\\hline
\\multirow{{4}}{{*}} {{ML-1M}}
{make_block("ml1m")}
\\hline
\\end{{tabular}}
\\label{{tab:results}}
\\end{{table*}}
"""

print(s)
