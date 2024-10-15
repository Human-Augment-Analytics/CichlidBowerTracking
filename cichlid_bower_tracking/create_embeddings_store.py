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

identities = train_df['identity'].unique()
train_subsets, valid_subsets = [], []
for identity in identities:
    subset = train_df[train_df['identity'] == identity].to_numpy()

    if subset.shape[0] == 1:
        continue

    train_subset, valid_subset = train_test_split(subset, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    train_subsets.append(train_subset)
    valid_subsets.append(valid_subset)

train_df = pd.DataFrame(np.row_stack(train_subsets), columns=train_df.columns)
valid_df = pd.DataFrame(np.row_stack(valid_subsets), columns=train_df.columns)

print(f'\nTraining Set %: {train_df.shape[0] / (test_df.shape[0] + train_df.shape[0] + valid_df.shape[0])}')
print(f'Validation Set %: {valid_df.shape[0] / (test_df.shape[0] + train_df.shape[0] + valid_df.shape[0])}')
print(f'Testing Set %: {test_df.shape[0] / (test_df.shape[0] + train_df.shape[0] + valid_df.shape[0])}')

print(f'\n% of original dataset: {(test_df.shape[0] + train_df.shape[0] + valid_df.shape[0]) / df.shape[0]}\n')

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