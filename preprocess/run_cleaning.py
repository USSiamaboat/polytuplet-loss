# Imports
import json
import pandas as pd
import numpy as np
import re
import string

# File Paths
TRAIN_FILE_PATH = "dataset/raw_data/train.json"
VAL_FILE_PATH = "dataset/raw_data/val.json"

# Load files into DataFrames
with open(TRAIN_FILE_PATH, "rb") as f:
  train_df = pd.DataFrame(json.loads(f.read()))
  del f

with open(VAL_FILE_PATH, "rb") as f:
  val_df = pd.DataFrame(json.loads(f.read()))
  del f

df = pd.concat([train_df, val_df])

# Generate split
df["split"] = df["id_string"].apply(lambda x: x.split("_")[0])

# Cleaning function
def clean(t):
  # Lowercase and remove punctuation
  t = t.lower()
  t = t.translate(str.maketrans('', '', string.punctuation))
  
  # removing extra space and letters
  t = re.sub("\s+", ' ', t)
  t = re.sub("\b\w\b", '', t)

  return t

# Apply cleaning function to text data
for column in ["context", "question"]:
  df[column] = df[column].apply(clean)

df["answers"] = df["answers"].apply(lambda x: [clean(a) for a in x])

# Put answers in individual columns
for i in range(4):
  df[f"answer_{i+1}"] = df["answers"].apply(lambda x: x[i])

# Remove unnecessary columns
df = df.drop(columns=["answers", "id_string"])

# Save
df.to_csv("dataset/cleaned/dev.csv", index=False)
