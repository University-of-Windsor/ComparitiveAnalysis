# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:57:20 2023

@author: aliab
"""
import tensorflow as tf
from tensorflow.keras import layers

def attack_model_fn():
    model = tf.keras.models.Sequential()
    model.add(layers.Dense(64, activation="relu", input_shape=(10,)))
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model