# Basics
import pandas as pd
import numpy as np
import os
import sys
from mock import patch

# tf
import tensorflow as tf

# Models
from transformers import (AlbertConfig, TFAlbertModel,
                          DistilBertConfig, TFDistilBertModel,
                          RobertaConfig, TFRobertaModel)

import sentencepiece as spm

from preprocess.preprocess import Preprocess
from models.polytuplet import PolytupletModel

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

# Ensure path exists
if not os.path.exists("dataset/processed"):
	os.makedirs("dataset/processed")

# Load cleaned data
df = pd.read_csv("dataset/cleaned/dev.csv")
print(df.head())

# Preprocess and save dataset-ready tuples for each model
for model_index in range(3):
	print(model_index)

	# Load preprocessor
	preprocessor = Preprocess(df=df, model_index=model_index)

	# Generate dataset-ready tuples
	train_data, val_data = preprocessor.get_datasets(mixed=USE_MIXED)

model = PolytupletModel(preprocessor.CONTEXT_LEN, preprocessor.RESULT_LEN)

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
