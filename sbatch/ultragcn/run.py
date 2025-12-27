import argparse
import subprocess
import sys

DATASETS = [
    "amazon_apps_for_android",
    "amazon_beauty",
    "amazon_home_and_kitchen",
    "amazon_musical_instruments",
    "amazon_ratings_baby",
    "amazon_ratings_cds_and_vinyl",
    "amazon_ratings_grocery_and_gourmet_food",
    "amazon_ratings_kindle_store",
    "amazon_sport_and_outdoors",
    "amazon_toy_and_games",
    "anime_ratings",
    "ciao_dvd",
    "citeulike_a",
    "citeulike_t",
    "movielens",
    "movietweetings", ## 16
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_idx", type=int, required=True,
                        help="Index of dataset (0..N-1)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g. random, svd, als, etc.)")
    parser.add_argument("--num_trials", type=int, default=60)

    args = parser.parse_args()

    if args.dataset_idx < 0 or args.dataset_idx >= len(DATASETS):
        raise ValueError(
            f"dataset_idx must be in [0, {len(DATASETS)-1}], "
            f"got {args.dataset_idx}"
        )

    dataset = DATASETS[args.dataset_idx]

    cmd = [
        "python3",
        "run_optuna.py",
        "--config_name", args.model,
        "--dataset", dataset,
        "--optuna_params", "ultragcn_ml1m",
        "--experiment_name", args.model + "_" + dataset,
        "--num_trials", str(args.num_trials),
        "--timeout", str(48 * 60 * 60),
    ]

    print("Running command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
