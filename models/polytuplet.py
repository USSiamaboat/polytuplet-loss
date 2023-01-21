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
                          RobertaConfig, TFRobertaModel,)

# TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver("local")
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices("TPU"))
strategy = tf.distribute.TPUStrategy(resolver)

class SemiHardPolytupletLoss(tf.keras.Model):
  def __init__(self, alpha, m, hard_w, aggregator="sum", name="SemiHardPolytupletLoss"):
    super(SemiHardPolytupletLoss, self).__init__(name=name)
    self.alpha = alpha
    self.m = m
    self.hard_w = hard_w

    self.aggregator = {
      "sum": tf.reduce_sum,
      "max": tf.reduce_max
    }[aggregator]

  def call(self, context, positive, negatives):
    # Compute the squared norm
    d_cp = tf.expand_dims(tf.reduce_sum(tf.square(tf.subtract(context, positive)), axis=1), 1)
    d_cn = tf.reduce_sum(tf.square(tf.subtract(tf.expand_dims(context, 1), negatives)), axis=2)

    # Calculate the difference between the norms
    d_diffs = tf.subtract(d_cp, d_cn)

    # Polytuplet loss
    polytuplet = self.aggregator(tf.nn.relu(tf.add(self.m, d_diffs)))

    # Create masking variables
    positive_closer = tf.less(d_diffs, 0.)
    bad_margin = tf.greater(polytuplet, 0.)

    # Create the semi-hard loss
    semihard_mask = tf.cast(tf.math.logical_and(positive_closer, bad_margin), dtype=tf.float32)
    semihard_mask_len = tf.add(tf.reduce_sum(semihard_mask), 1e-10)
    semihard_loss = tf.divide(tf.reduce_sum(tf.multiply(polytuplet, semihard_mask)), semihard_mask_len)

    # Create hard negative mining mask
    hard_mask = tf.cast(tf.math.logical_not(positive_closer), dtype=tf.float32)
    hard_mask_len = tf.add(tf.reduce_sum(hard_mask), 1e-10)
    hard_loss = tf.divide(tf.reduce_sum(tf.multiply(polytuplet, hard_mask)), hard_mask_len)
    hard_loss = tf.multiply(hard_loss, self.hard_w)

    # Calculate weighted loss
    return tf.multiply(tf.add(semihard_loss, hard_loss), self.alpha)

class DistanceClassifier(tf.keras.Model):
  def __init__(self, name="DistanceClassifier"):
    super(DistanceClassifier, self).__init__(name=name)

  def call(self, context_embeddings, result_embeddings):
    distances = tf.norm(tf.subtract(tf.expand_dims(context_embeddings, 1), result_embeddings), axis=2)
    activation = tf.keras.layers.Activation("softmax")(distances)
    probabilities = tf.subtract(1., activation)
    return probabilities

class Accuracy(tf.keras.Model):
  def __init__(self, name="Accuracy"):
    super(Accuracy, self).__init__(name=name)

  def call(self, classification):
    return tf.cast(tf.equal(tf.constant(3, dtype=tf.int8), tf.cast(tf.argmax(classification, axis=1), dtype=tf.int8)), dtype=tf.float16)

class Hypersphere(tf.keras.Model):
  def __init__(self, name="Hypersphere"):
    super(Hypersphere, self).__init__(name=name)

  def call(self, embeddings):
    normalized, norm = tf.linalg.normalize(embeddings, axis=1)
    return tf.where(tf.less(tf.abs(embeddings), tf.abs(normalized)), embeddings, normalized)

class PolytupletModel():
  def __init__(self, context_len, result_len, dropout=0.1, learning_rate=9e-6, final_learning_rate=3e-6, initializer_range=0.02, alpha=2, m=1.4, l2=0.5, model_index=0, seed=42, use_tpu=True):
    """
    Initlializes baseline model type
    
    Args:
      context_len (int): the length of the context embedding
      result_len (int): the length of the result embedding
      dropout (float): dropout rate
      learning_rate (float): initial learning rate
      final_learning_rate (float): the learning rate at the end of linear decay
      initializer_range (float): standard deviation of the kernel initializer
      alpha (float): the coefficient for polytuplet loss
      l2 (float): the value of 
      m (float): the polytuplet loss minimum margin
      seed (int): random seed
      model_index (0, 1, 2): selects albert, distilbert, or roberta models
    
    Returns:
      None

    """

    # Set hyperparameters
    self.dropout = dropout
    self.learning_rate = learning_rate
    self.final_learning_rate = final_learning_rate
    self.alpha = alpha
    self.l2 = l2
    self.m = m

    # Set other properties
    self.CONTEXT_LEN = context_len
    self.RESULT_LEN = result_len
    self.seed = seed
    self.kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=initializer_range, seed=self.seed)
    self.use_tpu = use_tpu
    self.hidden_size = 128
    self.embedding_size = 64
    self.batch_size = 16
    self.hard_w = 0.25

    # Shortcut name
    self.shortcut_name = (
                           "albert-xxlarge-v2",
                           "distilbert-base-cased",
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

    # ------------ PREPARATION ------------ #
    config = self.config
    # Load pretrained model for context processing
    context_pretrained = self.pretrained(config)
    context_pretrained = context_pretrained.from_pretrained(
                           self.shortcut_name,
                         )

    # Load pretrained model for result processing
    result_pretrained = self.pretrained(config)
    result_pretrained = result_pretrained.from_pretrained(
                          self.shortcut_name,
                        )

    # ------------ INPUTS ------------ #
    # Context inputs
    c_input_ids_ = tf.keras.Input(
                     shape=(self.CONTEXT_LEN, ),
                     dtype='int32',
                     batch_size=self.batch_size,
                     name="c_input_ids"
                   )
    c_masks_ = tf.keras.Input(
                 shape=(self.CONTEXT_LEN, ),
                 dtype='int32',
                 batch_size=self.batch_size,
                 name="c_attention_mask"
               )

    # Result inputs
    r_input_ids_ = tf.keras.Input(
                     shape=(4, self.RESULT_LEN),
                     dtype='int32',
                     batch_size=self.batch_size,
                     name="r_input_ids"
                   )
    r_masks_ = tf.keras.Input(
                 shape=(4, self.RESULT_LEN),
                 dtype='int32',
                 batch_size=self.batch_size,
                 name="r_attention_mask"
               )

    # ------------ CONTEXT ------------ #
    # Pass context into ALBERT and get last hidden layer
    context_hidden_states = context_pretrained(
                              input_ids=c_input_ids_,
                              attention_mask=c_masks_,
                              training=True
                            )[0]
    context_pooled = context_hidden_states[:, 0]
    context_pooled = tf.keras.layers.Dense(
                       768,
                       kernel_initializer=self.kernel_initializer,
                       activation="relu",
                       name="context_pre_classifier",
                     )(context_pooled)
    context_pooled = tf.keras.layers.Dropout(self.dropout)(context_pooled)

    # Convert last hidden layer of ALBERT to embedding
    c_hidden = tf.keras.layers.Dense(
                 self.hidden_size,
                 activation="relu",
                 kernel_initializer=self.kernel_initializer,
                 name="c_hidden"
               )(context_pooled)
    context_embeddings = tf.keras.layers.Dense(
                           self.embedding_size,
                           activation="tanh",
                           kernel_initializer=self.kernel_initializer,
                           kernel_constraint=tf.keras.constraints.UnitNorm(axis=1),
                           name="c_embedding"
                         )(c_hidden)
    context_embeddings = tf.linalg.normalize(
                           context_embeddings,
                           axis=1,
                           name="c_normalized"
                         )[0]

    # ------------ RESULT ------------ #
    # Reshape inputs into (batch_size*4, RESULT_LEN)
    flat_input_ids = tf.reshape(
                       r_input_ids_,
                       (-1, self.RESULT_LEN),
                       name="r_ids_flatten"
                     )
    flat_attention_mask = tf.reshape(
                            tensor=r_masks_,
                            shape=(-1, self.RESULT_LEN),
                            name="r_mask_flatten"
                          )

    # Pass context into pretrained and get last hidden layer
    result_hidden_states = result_pretrained(
                             input_ids=flat_input_ids,
                             attention_mask=flat_attention_mask,
                             training=True
                           )[0]
    result_pooled = result_hidden_states[:, 0]
    result_pooled = tf.keras.layers.Dense(
                      768,
                      kernel_initializer=self.kernel_initializer,
                      activation="relu",
                      name="result_pre_classifier",
                    )(result_pooled)
    result_pooled = tf.keras.layers.Dropout(self.dropout)(result_pooled)

    # Convert last hidden layer of pretrained to embedding
    r_hidden = tf.keras.layers.Dense(
                 self.hidden_size,
                 activation="relu",
                 kernel_initializer=self.kernel_initializer,
                 name="r_hidden"
               )(result_pooled)
    result_embeddings_flattened = tf.keras.layers.Dense(
                                    self.embedding_size,
                                    activation="tanh",
                                    kernel_initializer=self.kernel_initializer,
                                    kernel_constraint=tf.keras.constraints.UnitNorm(axis=1),
                                    name="r_embedding"
                                  )(r_hidden)
    result_embeddings_flattened = tf.linalg.normalize(
                                    result_embeddings_flattened,
                                    axis=1,
                                    name="r_normalized"
                                  )[0]

    # Reshape inputs back to (batch_size, 4, embedding_size)
    result_embeddings = tf.reshape(
                          result_embeddings_flattened,
                          (-1, 4, self.embedding_size),
                          "r_reshape"
                        )

    # ------------ CLASSIFICATION ------------ #
    # Compute distances
    classification = DistanceClassifier()(context_embeddings, result_embeddings)

    # Make model
    model = tf.keras.Model(inputs=[c_input_ids_, c_masks_, r_input_ids_, r_masks_], outputs=classification, name="ALBERT_triplet")

    # ------------ LOSS ------------ #
    # Create temporary variables for loss calculation
    contexts = context_embeddings
    positives = result_embeddings[:, 3, :]
    negatives = result_embeddings[:, 0:3, :]

    model.add_loss(SemiHardPolytupletLoss(alpha=self.alpha, m=self.m, hard_w=self.hard_w, aggregator="sum")(contexts, positives, negatives))

    # ------------ METRICS ------------ #
    # Compute Batch Accuracy
    model.add_metric(Accuracy()(classification), name="accuracy")


    # ------------ COMPILE ------------ #
    # Generate Schedule
    schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                 self.learning_rate, 5, self.final_learning_rate, 1
               )

    optimizer = tf.keras.optimizers.Adam(learning_rate=schedule, epsilon=1e-6, beta_1=0.9, beta_2=0.98)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy")
    
    
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
    Fit model to data using semi-hard mining, updates model property
    
    Args:
      train_data (tf.data.Dataset): data to train on
      validation_data (tf.data.Dataset): data to validate with
      epochs (int): the maximum number of epochs to train for
      early_stopping (boolean): whether the model should train for maximum epochs
                                or stop when validation loss stops improving for
                                patience epochs
      patience (int): the number of epochs to wait before stopping training, only
                      in effect when early_stopping is True
      verbose (int | string): passthrough to tensorflow model .fit verbose
    
    Returns:
      (keras.callbacks.History): the history object from fitting the model

    """

    # Check if model exists
    if not self.model:
      raise AttributeError("Model must be defined to train. Run .compile before .fit")
    
    # Adjust parameters
    if not early_stopping:
      patience = epochs

    # Set callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=patience, restore_best_weights=True)

    # Set kwargs
    kwargs = {
      "x": train_data.batch(self.batch_size, drop_remainder=True),
      "validation_data": validation_data.batch(self.batch_size, drop_remainder=True),
      "callbacks": [early_stopping],
      "verbose": verbose,
      "epochs": epochs
    }

    self.model.fit(**kwargs)

  def tune_hyperparams(self, train_data, validation_data, dropout_range, learning_rate_range, final_learning_rate_range, alpha_range, m_range, hard_w_range):
    """
    Tunes model hyperparamters within given ranges
    
    Args:
      train_data (tf.data.Dataset): data to train on
      validation_data (tf.data.Dataset): data to validate with
      dropout_range (float, float): the upper and lower bounds for the dropout search range
      learning_rate_range (float, float): the upper and lower bounds for the learning rate search range
      final_learning_rate_range (float, float): the upper and lower bounds for the final learning rate search range
      alpha_range (float, float): the upper and lower bounds for the coefficient of polytuplet loss
      m_range (float, float): the upper and lower bounds for the polytuplet loss minimum margin search range
      hard_w_range (float, float): the upper and lower bounds for the weighting of hard negatives in polytuplet loss

    
    Returns:
      (kerastuner.Hyperband): the history object from fitting the model

    """

    # Container function for building model during hyperparameter tuning
    def build_model(hp):
      self.dropout = hp.Float("dropout", *dropout_range)
      self.learning_rate = hp.Float("learning_rate", *learning_rate_range, sampling="log")
      self.final_learning_rate = hp.Float("final_learning_rate", *final_learning_rate_range, sampling="log")
      self.alpha = hp.Float("alpha", *alpha_range)
      self.m = hp.Float("m", *m_range)
      self.hard_w = hp.Float("hard_w", *hard_w_range)

			# Reset TPU
      tf.tpu.experimental.initialize_tpu_system(resolver)

      return self.build_()

    tuner = kt.Hyperband(
              build_model,
              objective='val_accuracy',
              max_epochs=30,
              hyperband_iterations=1,
              distribution_strategy=strategy,
              seed=self.seed,
              overwrite=True
            )

    tuner.search(
      train_data.batch(self.batch_size, drop_remainder=True),
      validation_data=validation_data.batch(self.batch_size, drop_remainder=True),
      epochs=30,
      callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=1, restore_best_weights=True)]
    )

    return tuner.get_best_hyperparameters()[0].values