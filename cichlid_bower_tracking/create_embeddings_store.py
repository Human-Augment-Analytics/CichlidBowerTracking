from typing import Dict, List

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import json

BASE = '/home/hice1/cclark339/scratch/Data/WildlifeReID-10K/'
df = pd.read_csv(BASE + 'metadata.csv')

train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

RANDOM_STATE = 42
TEST_SIZE = 0.1
train_df, valid_df = train_test_split(train_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

train_df.to_csv(BASE + 'train_metadata.csv')
valid_df.to_csv(BASE + 'valid_metadata.csv')
test_df.to_csv(BASE + 'test_metadata.csv')

del df

def gen_embeddings_store(data: pd.DataFrame) -> Dict[str, Dict[str, List[float]]]:
    embeddings_store = dict()

    unique_ids = data['identity'].unique()
    for unique_id in unique_ids:
        subset = data[data['identity'] == unique_id]

        embeddings_store[unique_id] = {path: np.random.randn(5).tolist() for path in subset['path'].to_list()}

    return embeddings_store

def save_embeddings_store(embeddings_store: Dict[str, Dict[str, List[float]]]) -> None:
    with open(BASE + 'train_embeddings_store.json', 'w') as outfile:
        json.dump(embeddings_store, outfile)

train_embeddings_store = gen_embeddings_store(train_df)
save_embeddings_store(train_embeddings_store)

print(f'Initial training embeddings store saved to "{BASE + "train_embeddings_store.json"}"')