# Basics
import pandas as pd
import numpy as np

# tf
import tensorflow as tf

# Models
from transformers import (AlbertTokenizer,
                          DistilBertTokenizer,
                          RobertaTokenizer,
                          BartTokenizer)

import sentencepiece as spm

class Preprocess():
  def __init__(self, df, model_index=0):
    """
    Initlializes data preprocessor
    
    Args:
      df (pd.DataFrame): A DataFrame with split, context, question, answer_1, answer_2, answer_3, and answer_4 columns
      model_index (0, 1, 2, 3): selects albert, distilbert, distilroberta, or bart models
    
    Returns:
      None

    """

    self.df = df
    self.CONTEXT_LEN = 0
    self.RESULT_LEN = 0

    # Shortcut name
    self.shortcut_name = (
                           "albert-large-v2",
                           "distilbert-base-cased",
                           "distilroberta-base",
                           "facebook/bart-base"
                         )[model_index]

    # Tokenizer
    self.tokenizer = (
                       AlbertTokenizer,
                       DistilBertTokenizer,
                       RobertaTokenizer,
                       BartTokenizer
                     )[model_index]
    self.tokenizer = self.tokenizer.from_pretrained(self.shortcut_name)
  
  def split(self, df):
    """
    Finds the starting index of validation data
    
    Args:
      df (pd.DataFrame): the data to split
    
    Returns:
      (int): the first index of validation data

    """

    return np.where(df["split"] == "val")[0][0]
  
  def longest_tokenized(self, vector):
    """
    Determines the maximum tokenized length of multiple strings
    
    Args:
      vector (np.ndarray[String]): A one-dimensional vector of strings
    
    Returns:
      (int): the maximum tokenized length of any string in the input vector

    """
    max_len = 0
    for string in vector:
      max_len = max(max_len, len(self.tokenizer(string)["input_ids"]))
    return max_len
  
  def get_datasets(self, mixed=True):
    """
    Generates TensorFlow datasets for training and validation
    
    Args:
      mixed (boolean): Whether to mix the index of the correct answer
    
    Returns:
      (tf.data.Dataset tuple, tf.data.Dataset tuple): a tuple of dataset-ready tuples, ordered train split then test split

    """

    # Create results columns
    for answer_col_i in range(1, 5):
      self.df[f"result_{answer_col_i}"] = self.df[["question", f"answer_{answer_col_i}"]].apply(" ".join, axis=1)
    
    # Determine maximum tokenized length for context if needed
    if not self.CONTEXT_LEN:
      self.CONTEXT_LEN = self.longest_tokenized(self.df["context"].to_numpy())

    # Generate labels
    mixed_labels = np.zeros((len(self.df), 4))
    mixed_labels[np.arange(len(self.df)), self.df["label"].to_numpy()] = 1

    unmixed_labels = np.zeros((len(self.df), 4))
    unmixed_labels[:, -1] = 1

    if mixed:
      labels = mixed_labels
    elif not mixed:
      labels = unmixed_labels

    # Reorder results if not mixed
    if not mixed:
      results = self.df[[f"result_{i}" for i in range(1, 5)]].to_numpy()
      positive_results = results[mixed_labels.astype("bool")].reshape(-1, 1)
      negative_results = results[(1-mixed_labels).astype("bool")].reshape(-1, 3)
      results = np.concatenate((negative_results, positive_results), axis=1)

      self.df[[f"result_{i}" for i in range(1, 5)]] = results

    # Tokenize contexts and results
    tokenized_contexts = [
      self.tokenizer(context, max_length=self.CONTEXT_LEN, padding="max_length")
      for context in self.df["context"].to_numpy()
    ]

    tokenized_results = self.tokenizer(
      np.array([[context]*4 for context in self.df["context"]]).flatten().tolist(),
      self.df[[f"result_{i}" for i in range(1, 5)]].to_numpy().flatten().tolist(),
      padding="longest"
    )

    # Set RESULT_LEN
    self.RESULT_LEN = np.array(tokenized_results["input_ids"]).reshape(len(self.df), 4, -1).shape[-1]

    # Prepare for dataset conversion
    data = {
      "c_input_ids": np.array([context["input_ids"] for context in tokenized_contexts]),
      "c_attention_mask": np.array([context["attention_mask"] for context in tokenized_contexts]),
      "r_input_ids": np.array(tokenized_results["input_ids"]).reshape(len(self.df), 4, -1),
      "r_attention_mask": np.array(tokenized_results["attention_mask"]).reshape(len(self.df), 4, -1)
    }

    # Split data and labels on train/val
    train_val_index = self.split(self.df)

    train_data = {k: v[:train_val_index] for k, v in data.items()}
    val_data = {k: v[train_val_index:] for k, v in data.items()}

    train_labels = labels[:train_val_index]
    val_labels = labels[train_val_index:]

    # Generate dataset-ready tuples
    train = (train_data, train_labels)
    val = (val_data, val_labels)

    return (train, val)
