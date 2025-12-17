import argparse
import subprocess
import sys

DATASETS = [
    "amazon_apps_for_android",                          ## 0
    "amazon_beauty",                                    ## 1
    "amazon_home_and_kitchen",                          ## 2
    "amazon_musical_instruments",                       ## 3
    "amazon_ratings_baby",                              ## 4
    "amazon_ratings_cds_and_vinyl",                     ## 5
    "amazon_ratings_grocery_and_gourmet_food",          ## 6
    "amazon_ratings_kindle_store",                      ## 7
    "amazon_sport_and_outdoors",                        ## 8
    "amazon_toy_and_games",                             ## 9
    "anime_ratings",                                    ## 10
    "ciao_dvd",                                         ## 11
    "citeulike_a",                                      ## 12
    "citeulike_t",                                      ## 13
    "movielens",                                        ## 14
    "movietweetings",                                   ## 15
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_idx", type=int, required=True,
                        help="Index of dataset (0..N-1)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g. random, svd, als, etc.)")

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
        "--optuna_params", "pure_svd",
        "--experiment_name", args.model + "_" + dataset,
    ]

    print("Running command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
