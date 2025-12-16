from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path

from src.utils.download import download


for dataset_cfg in tqdm(Path("configs/dataset").iterdir(), desc="Downloading datasets...", total=sum([1 for _ in Path("configs/dataset").iterdir()])):
    yaml = OmegaConf.load(dataset_cfg)
    download(yaml.get('url'), f"data/{yaml.get('name')}")
