#!/bin/bash

# Submit all GASATF optuna jobs

# Dense datasets (high interaction density)
sbatch sbatch/gasatf/movielens.sbatch
sbatch sbatch/gasatf/anime_ratings.sbatch

# Medium density datasets
sbatch sbatch/gasatf/ciao_dvd.sbatch
sbatch sbatch/gasatf/movietweetings.sbatch

# Sparse datasets (Amazon, CiteULike)
sbatch sbatch/gasatf/amazon_beauty.sbatch
sbatch sbatch/gasatf/amazon_apps_for_android.sbatch
sbatch sbatch/gasatf/amazon_home_and_kitchen.sbatch
sbatch sbatch/gasatf/amazon_musical_instruments.sbatch
sbatch sbatch/gasatf/amazon_ratings_baby.sbatch
sbatch sbatch/gasatf/amazon_ratings_cds_and_vinyl.sbatch
sbatch sbatch/gasatf/amazon_ratings_grocery_and_gourmet_food.sbatch
sbatch sbatch/gasatf/amazon_ratings_kindle_store.sbatch
sbatch sbatch/gasatf/amazon_sport_and_outdoors.sbatch
sbatch sbatch/gasatf/amazon_toy_and_games.sbatch
sbatch sbatch/gasatf/citeulike_a.sbatch
sbatch sbatch/gasatf/citeulike_t.sbatch

echo "All GASATF jobs submitted!"
