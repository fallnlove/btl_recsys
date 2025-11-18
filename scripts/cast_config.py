import json

from omegaconf import OmegaConf


def cast_config(config_name):
    with open(f"configs/{config_name}.json", "r") as f:
        config = json.load(f)

    with open(f"configs/{config_name}.yaml", "w") as f:
        print(OmegaConf.to_yaml(config), file=f)


for config_name in ["beauty", "clothes", "ml1m", "sasrec_grid", "sports"]:
    cast_config(config_name)
