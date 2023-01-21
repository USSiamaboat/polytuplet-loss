# Basics
import pandas as pd
import numpy as np
import os
import sys

# tf
import tensorflow as tf

# Models
from transformers import (AlbertConfig, TFAlbertModel,
                          DistilBertConfig, TFDistilBertModel,
                          RobertaConfig, TFRobertaModel)

import sentencepiece as spm

from preprocess.preprocess import Preprocess
from models.baseline import BaselineModel

# Tuning
import keras_tuner as kt

# Get system arguments
model_name_map = {
	"albert": 0,
	"distilbert": 1,
	"roberta": 2
}
MODEL_INDEX = model_name_map[sys.argv[1]]
USE_MIXED = sys.argv[2] == "mixed"

# for windows
if os.name == 'nt':
	os.system('cls')
else:
	os.system('clear')

print("="*20)
print("Running trainer with config")
print(f"Model name {sys.argv[1]} and index {MODEL_INDEX}")
print(f"Data mixing is {USE_MIXED}")
print("="*20)

# Ensure path exists
if not os.path.exists("dataset/processed"):
	os.makedirs("dataset/processed")

# Load cleaned data
print("Loading data...")
df = pd.read_csv("dataset/cleaned/dev.csv")
print("Data loaded")

# Load preprocessor
print("Preprocessing...")
preprocessor = Preprocess(df=df, model_index=MODEL_INDEX)

# Generate dataset-ready tuples
train_data, val_data = preprocessor.get_datasets(mixed=USE_MIXED)

train_data = ({"input_ids": train_data[0]["r_input_ids"], "attention_mask": train_data[0]["r_attention_mask"]}, train_data[1])
val_data = ({"input_ids": val_data[0]["r_input_ids"], "attention_mask": val_data[0]["r_attention_mask"]}, val_data[1])

train_data = tf.data.Dataset.from_tensor_slices(train_data)
val_data = tf.data.Dataset.from_tensor_slices(val_data)
print("Preprocessing complete")

print("Building model")
model = BaselineModel(preprocessor.RESULT_LEN)

model.tune_hyperparams(
  train_data=train_data,
  validation_data=val_data,
  dropout_range=(0.0, 0.3),
  learning_rate_range=(1e-7, 1e-5),
  final_learning_rate_range=(1e-7, 1e-5),
  alpha_range=(0.0, 10.0),
  m_range=(0.0, 2.0),
  hard_w_range=(0.0, 1.0)
)
