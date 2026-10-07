#!/usr/bin/env python3
"""
A tiny CNN with the same layer layout as the HAR model, only much smaller.
Weights are random: this model exists to test the toolchain and the boards,
not to classify anything. Same structure as HAR means any difference between
the two on hardware comes down to size.

Writes:
  tiny_cnn.keras          the Keras model
  savedmodel_tiny/        float SavedModel, for the Q8.8 path
  tiny_cnn_int8.tflite    INT8 TFLite, for the CPU path
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

L, C = 64, 3
init_b = keras.initializers.RandomNormal(0.0, 0.1, seed=1)

def conv(f, k, s):
    return keras.layers.Conv1D(f, k, strides=s, padding="same",
                               activation="relu", bias_initializer=init_b)

keras.utils.set_random_seed(0)

inp = keras.Input(shape=(L, C))
x = conv(4, 5, 2)(inp)
x = keras.layers.MaxPooling1D(2)(x)
x = conv(8, 3, 2)(x)
x = keras.layers.MaxPooling1D(2)(x)
x = conv(8, 3, 1)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dense(8, activation="relu", bias_initializer=init_b)(x)
out = keras.layers.Dense(4, bias_initializer=init_b)(x)

model = keras.Model(inp, out)
model.summary()

model.save("tiny_cnn.keras")
model.export("savedmodel_tiny")

rng = np.random.default_rng(0)

def representative():
    for _ in range(64):
        yield [rng.normal(0, 0.5, (1, L, C)).astype(np.float32)]

conv_tfl = tf.lite.TFLiteConverter.from_keras_model(model)
conv_tfl.optimizations = [tf.lite.Optimize.DEFAULT]
conv_tfl.representative_dataset = representative
conv_tfl.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv_tfl.inference_input_type = tf.int8
conv_tfl.inference_output_type = tf.int8
open("tiny_cnn_int8.tflite", "wb").write(conv_tfl.convert())

print("wrote tiny_cnn.keras, savedmodel_tiny/, tiny_cnn_int8.tflite")