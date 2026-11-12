#!/usr/bin/env python3
# Copyright 2026 Indonesia Chip Design Collaborative Center (ICDeC)
# Author: Muhammad Faudzan Abdullah
# SPDX-License-Identifier: Apache-2.0

"""
Build a tiny 1D-CNN that uses depthwise convolution and a residual add, then
quantize it to INT8. Weights are random: this model exists to exercise the
importer, not to classify anything.

    python3 make_dw_residual.py
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

L, C = 128, 3

inp = keras.Input(shape=(L, C))
x = keras.layers.Conv1D(8, 5, padding="same", activation="relu")(inp)

# depthwise block with a residual connection around it
y = keras.layers.DepthwiseConv1D(3, padding="same", activation="relu")(x)
y = keras.layers.Conv1D(8, 1, padding="same")(y)          # pointwise
x = keras.layers.Add()([x, y])
x = keras.layers.ReLU()(x)

x = keras.layers.MaxPooling1D(2)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
out = keras.layers.Dense(4)(x)

model = keras.Model(inp, out)
model.summary()

rng = np.random.default_rng(0)


def representative():
    for _ in range(64):
        yield [rng.normal(0, 1, (1, L, C)).astype(np.float32)]


conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = representative
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8

open("dw_residual_int8.tflite", "wb").write(conv.convert())
print("wrote dw_residual_int8.tflite")