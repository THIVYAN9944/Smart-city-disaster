"""
Smart City Disaster - Deep Learning Models
==========================================
Implements:
  - MLP (multi-layer perceptron) with dropout + batch-norm
  - 1D-CNN over feature embeddings
  - LSTM on reshaped sequences

The reshape for the sequence-style models treats each row of features as
a "time step" of width 1, so a 25-dim feature vector becomes (25, 1) -
this is a common technique when there is no real temporal ordering
in the data but we still want to demonstrate RNN/CNN-1D pipelines.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers


# Reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


def build_mlp(input_dim, num_classes,
              hidden_layers=(256, 128, 64),
              dropout=0.4, l2=1e-4, learning_rate=1e-3):
    """Multi-layer perceptron with batch-norm + dropout + L2 regularization."""
    model = models.Sequential(name='MLP')
    model.add(layers.Input(shape=(input_dim,)))
    for h in hidden_layers:
        model.add(layers.Dense(h, kernel_regularizer=regularizers.l2(l2)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def build_cnn1d(input_dim, num_classes,
                filters=(64, 128), kernel_size=3,
                dropout=0.4, l2=1e-4, learning_rate=1e-3):
    """1D-CNN treating features as a 1D sequence of length `input_dim`."""
    inp = layers.Input(shape=(input_dim, 1))
    x = inp
    for f in filters:
        x = layers.Conv1D(f, kernel_size=kernel_size, padding='same',
                          kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2, padding='same')(x)
        x = layers.Dropout(dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out, name='CNN1D')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def build_lstm(input_dim, num_classes,
               lstm_units=(128, 64), dropout=0.4, l2=1e-4, learning_rate=1e-3):
    """Stacked LSTM with attention pooling."""
    inp = layers.Input(shape=(input_dim, 1))
    x = inp
    for i, u in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)
        x = layers.LSTM(u, return_sequences=return_sequences,
                        dropout=dropout,
                        kernel_regularizer=regularizers.l2(l2),
                        recurrent_regularizer=regularizers.l2(l2))(x)
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out, name='LSTM')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def build_transformer(input_dim, num_classes,
                      embed_dim=64, num_heads=4, ff_dim=128,
                      dropout=0.4, l2=1e-4, learning_rate=1e-3):
    """Lightweight Transformer encoder for tabular data."""
    inp = layers.Input(shape=(input_dim,))
    x = layers.Reshape((input_dim, 1))(inp)
    x = layers.Dense(embed_dim)(x)
    attn_in = layers.LayerNormalization(epsilon=1e-6)(x)
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads,
        dropout=dropout)(attn_in, attn_in)
    attn_out = layers.Dropout(dropout)(attn_out)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
    ff = layers.Dense(ff_dim, activation='relu',
                      kernel_regularizer=regularizers.l2(l2))(x)
    ff = layers.Dense(embed_dim)(ff)
    ff = layers.Dropout(dropout)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out, name='Transformer')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def get_callbacks(model_name, patience=8):
    return [
        callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=4, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(
            filepath=f'models/best_{model_name}.h5',
            monitor='val_accuracy', save_best_only=True, verbose=0),
    ]


def reshape_for_seq(X):
    """(N, F) -> (N, F, 1)"""
    return X.reshape((X.shape[0], X.shape[1], 1))
