dataset_list=(
    "amazon_apps_for_android"
    "amazon_beauty"
    "amazon_home_and_kitchen"
    "amazon_musical_instruments"
    "amazon_ratings_baby"
    "amazon_ratings_cds_and_vinyl"
    "amazon_ratings_grocery_and_gourmet_food"
    "amazon_ratings_kindle_store"
    "amazon_ratings_sport_and_outdoors"
    "amazon_ratings_toy_and_games"
    "anime_ratings"
    "ciao_dvd"
    "citeulike_a"
    "citeulike_t"
    "movielens"
    "movietweetings"
)

for i in ${dataset_list[@]}; do
  python run_optuna.py -cn ultragcn -ds $i -op ultragcn_ml1m -en ultragcn_$i
done
