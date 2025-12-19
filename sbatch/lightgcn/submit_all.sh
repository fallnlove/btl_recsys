#!/bin/bash

# Submit all LightGCN optuna jobs

# Dense datasets (high interaction density)
sbatch sbatch/lightgcn/movielens.sbatch
sbatch sbatch/lightgcn/anime_ratings.sbatch

# Medium density datasets
sbatch sbatch/lightgcn/ciao_dvd.sbatch
sbatch sbatch/lightgcn/movietweetings.sbatch

# Sparse datasets (Amazon, CiteULike)
# sbatch sbatch/lightgcn/amazon_beauty.sbatch
# sbatch sbatch/lightgcn/amazon_apps_for_android.sbatch
# sbatch sbatch/lightgcn/amazon_home_and_kitchen.sbatch
# sbatch sbatch/lightgcn/amazon_musical_instruments.sbatch
# sbatch sbatch/lightgcn/amazon_ratings_baby.sbatch
# sbatch sbatch/lightgcn/amazon_ratings_cds_and_vinyl.sbatch
# sbatch sbatch/lightgcn/amazon_ratings_grocery_and_gourmet_food.sbatch
# sbatch sbatch/lightgcn/amazon_ratings_kindle_store.sbatch
# sbatch sbatch/lightgcn/amazon_sport_and_outdoors.sbatch
# sbatch sbatch/lightgcn/amazon_toy_and_games.sbatch
sbatch sbatch/lightgcn/citeulike_a.sbatch
sbatch sbatch/lightgcn/citeulike_t.sbatch

echo "All LightGCN jobs submitted!"
