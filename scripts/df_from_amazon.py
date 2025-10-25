import argparse
import pandas as pd
import gzip
import numpy as np
from pathlib import Path


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--file_name', type=str)

  args = parser.parse_args()

  df = getDF(args.file_name)
  

  new_df = df[["reviewerID", "asin", "overall", "unixReviewTime"]].copy()
  new_df.columns = ["user_id", "item_id", "rating", "timestamp"]
  new_df["rating"] = 1

  new_df["user_id"], unique_user_ids = pd.factorize(new_df["user_id"])
  new_df["item_id"], unique_item_ids = pd.factorize(new_df["item_id"])
  new_df["user_id"] += 1
  new_df["item_id"] += 1

  print(f"Final df shape: {new_df.shape}")
  print(f"First few rows:\n{new_df.head()}")

  print('saving...')
  Path("../data2").mkdir(exist_ok=True)
  new_df.to_csv("../data2/clothing.csv", index=False)
  print('saved successfully!')
