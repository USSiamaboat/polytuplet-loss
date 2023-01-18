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

from models.polytuplet import PolytupletModel

# Pickle
import pickle

# Tuning
import keras_tuner as kt

model_name_map = {
	"albert": 0,
	"distilbert": 1,
	"roberta": 2
}

MODEL_INDEX = model_name_map[sys.argv[1]]

model = PolytupletModel()

use_mixed = {"mixed": "mixed", "sorted": ""}[sys.argv[2]]

with open(f"model_{MODEL_INDEX}_datasets.pkl", "rb") as f:
	datasets = pickle.load(f)
	train_data = datasets["train"]
	val_data = datasets[f"{use_mixed}_val"]

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
