# Basics
import pandas as pd
import numpy as np

# tf
import tensorflow as tf

# Models
from transformers import (AlbertTokenizer,
                          DistilBertTokenizer,
                          RobertaTokenizer)

import sentencepiece as spm

from preprocess import Preprocess

# Pickle
import pickle

# Load cleaned data
df = pd.read_csv("dataset/cleaned/dev.csv")

# Preprocess and save dataset-ready tuples for each model
for model_index in range(3):
	print(model_index)

	# Load preprocessor
	preprocessor = Preprocess(df=df, model_index=model_index)

	# Generate dataset-ready tuples
	train, val = preprocessor.get_datasets(mixed=False)
	mixed_train, mixed_val = preprocessor.get_datasets(mixed=True)

	# Compile
	datasets = {
		"train": train,
		"val": val,
		"mixed_train": mixed_train,
		"mixed_val": mixed_val
	}

	with open(f"../dataset/processed/model_{model_index}_datasets.pkl", 'wb') as output:
		pickle.dump(datasets, output, pickle.HIGHEST_PROTOCOL)