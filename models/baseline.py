# Basics
import numpy as np
from mock import patch

# tf
import tensorflow as tf

# Tuning
import keras_tuner as kt

# Albert
from transformers import (AlbertConfig, TFAlbertModel,
                          DistilBertConfig, TFDistilBertModel,
                          RobertaConfig, TFRobertaModel)

# TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver("local")
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices("TPU"))
strategy = tf.distribute.TPUStrategy(resolver)

class BaselineModel():
  def __init__(self, embedding_size=267, dropout=0.1, learning_rate=3e-6, initializer_range=0.02, model_index=0, seed=42, use_tpu=True):
    """
    Initlializes baseline model type
    
    Args:
      embedding_size (int): the size of the embeddings fed into the model
      dropout (float): dropout rate
      learning_rate (float, tf.keras.optimizers.schedules.LearningRateSchedule): initial learning rate or learning rate schedule
      initializer_range (float): standard deviation of the kernel initializer
      seed (int): random seed
      model_index (0, 1, 2): selects albert, distilbert, or roberta models
    
    Returns:
      None

    """

    # Set hyperparameters
    self.dropout = dropout
    self.learning_rate = learning_rate

    # Set other properties
    self.EMBEDDING_SIZE = embedding_size
    self.kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
    self.seed = seed
    self.use_tpu = use_tpu

    # Shortcut name
    self.shortcut_name = (
                           "albert-xxlarge-v2",
                           "distilbert-base-uncased",
                           "roberta-large",
                         )[model_index]

    # Config
    self.config = (
                    AlbertConfig,
                    DistilBertConfig,
                    RobertaConfig,
                  )[model_index]
    self.config = self.config(hidden_dropout_prob=self.dropout)

    self.pretrained = (
                        TFAlbertModel,
                        TFDistilBertModel,
                        TFRobertaModel,
                      )[model_index]
    
    # Others
    self.optimizer = None
    self.model = None
  
  def reset_random(self):
    tf.keras.utils.set_random_seed(self.seed)   
    tf.config.experimental.enable_op_determinism()
  
  def build_(self):
    """
    Internal method to build and compile model
    
    Args:
      None
    
    Returns:
      (tf.keras.Model): the model built and compiled

    """

    # Config and load pretrained model
    config = self.config
    pretrained = self.pretrained(config)
    pretrained = pretrained.from_pretrained(
                   self.shortcut_name,
                   name=self.shortcut_name)

    # Create input layers
    input_ids_ = tf.keras.Input(
                   shape = (4, self.EMBEDDING_SIZE),
                   dtype = 'int32',
                   name="input_ids"
                 )
    masks_ = tf.keras.Input(
               shape = (4, self.EMBEDDING_SIZE),
               dtype = 'int32',
               name="attention_mask")

    # Quadruple batch size
    flat_input_ids = tf.reshape(input_ids_, (-1, self.EMBEDDING_SIZE))
    flat_attention_mask = tf.reshape(tensor=masks_, shape=(-1, self.EMBEDDING_SIZE))

    # Use pretrained to get embedding
    hidden_states = pretrained(
                      input_ids=flat_input_ids,
                      attention_mask=flat_attention_mask,
                      training=True
                    )[0]
    pooled_output = hidden_states[:, 0]
    pooled_output = tf.keras.layers.Dense(
                      768,
                      kernel_initializer=self.kernel_initializer,
                      activation="relu",
                      name="pre_classifier",
                    )(pooled_output)
    pooled_output = tf.keras.layers.Dropout(self.dropout)(pooled_output)

    # MLP Classification Head
    classification = tf.keras.layers.Dense(
                       1,
                       kernel_initializer=self.kernel_initializer,
                       activation=None,
                       name="classification"
                     )(pooled_output)
    logits = tf.reshape(classification, (-1, 4))

    logits = tf.keras.layers.Activation("softmax")(logits)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(
                  learning_rate=self.learning_rate,
                  epsilon=1e-6,
                  beta_1=0.9,
                  beta_2=0.98
                )

    # Make model
    model = tf.keras.Model(
              inputs=[input_ids_, masks_],
              outputs=logits,
              name=f"{self.shortcut_name}_baseline"
            )
    model.compile(optimizer = optimizer,
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    
    # Set properties
    self.optimier = optimizer
    self.model = model

    # Return
    return model
  
  def build(self):
    """
    Build and compile model
    
    Args:
      None
    
    Returns:
      (tf.keras.Model): the model built and compiled

    """

    self.reset_random()

    if self.use_tpu:
      with strategy.scope():
        return self.build_()
    
    return self.build_()

  def fit(self, train_data, validation_data, epochs=10, early_stopping=True, patience=3, verbose="auto"):
    """
    Fit model to data, updates model property
    
    Args:
      train_data (tf.data.Dataset): data to train on
      validation_data (tf.data.Dataset): data to validate with
      epochs (int): the maximum number of epochs to train for
      early_stopping (boolean): whether the model should train for maximum epochs
                                or stop when validation loss stops improving for
                                patience epochs
      patience (int): the number of epochs to wait before stopping training, only
                      in effect when early_stopping is True
      verbose (boolean): whether to display information while training
    
    Returns:
      (keras.callbacks.History): the history object from fitting the model

    """

    # Check if model exists
    if not self.model:
      raise AttributeError("Model must be defined to train. Run .compile before .fit")
    
    # Adjust parameters
    if not early_stopping:
      patience = epochs
    
    if verbose:
      verbose = 2
    else:
      verbose = 0
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=patience, restore_best_weights=True)
    return self.model.fit(x=train_data.batch(16),
                          validation_data=validation_data.batch(16),
                          epochs=epochs,
                          shuffle=True,
                          callbacks=[callback],
                          verbose=verbose)

  def tune_hyperparams(self, train_data, validation_data, dropout_range, learning_rate_range):
    """
    Tunes model hyperparamters within given ranges
    
    Args:
      train_data (tf.data.Dataset): data to train on
      validation_data (tf.data.Dataset): data to validate with
      dropout_range (float, float): The upper and lower bounds for the dropout search range
      learning_rate_range (float, float): The upper and lower bounds for the learning rate search range
    
    Returns:
      (kerastuner.Hyperband): the history object from fitting the model

    """

    # Container function for building model during hyperparameter tuning
    def build_model(hp):
      self.dropout = hp.Float("dropout", *dropout_range)
      self.learning_rate = hp.Float("learning_rate", *learning_rate_range, sampling="log")
      tf.tpu.experimental.initialize_tpu_system(resolver)
      return self.build_()
    
    # Replacement function for on_epoch_end that doesn't save models
    def new_on_epoch_end(self, epoch, logs=None):
      if not self.objective.has_value(logs):
        return
      current_value = self.objective.get_value(logs)
      if self.objective.better_than(current_value, self.best_value):
        self.best_value = current_value

    tuner = kt.Hyperband(
              build_model,
              objective='val_accuracy',
              max_epochs=30,
              hyperband_iterations=1,
              distribution_strategy=strategy,
              seed=self.seed,
              overwrite=True,
            )

    with patch('keras_tuner.engine.tuner_utils.SaveBestEpoch.on_epoch_end', new_on_epoch_end):
      tuner.search(
        train_data.batch(16),
        validation_data=validation_data.batch(16),
        epochs=30,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=1, restore_best_weights=True)]
      )
    
    return tuner.get_best_hyperparameters()[0].values