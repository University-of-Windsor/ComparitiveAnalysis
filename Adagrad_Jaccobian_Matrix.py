# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 13:37:21 2023

@author: aliab
"""

import numpy as np
import tensorflow as tf
import random
import datetime
import tensorflow_privacy
import matplotlib.pyplot as plt
import pandas as pd

tf.random.set_seed(42)
np.random.seed(42)

def load_mnist():
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.mnist.load_data()
  #train, test = tf.keras.datasets.fashion_mnist.load_data()
  #train, test = tf.keras.datasets.cifar10.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255

  train_data = train_data.reshape((train_data.shape[0], 28, 28, 1))
  test_data = test_data.reshape((test_data.shape[0], 28, 28, 1))
  #train_data = train_data.reshape((train_data.shape[0], 32, 32, 3))
  #test_data = test_data.reshape((test_data.shape[0], 32, 32, 3))

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  #train_labels = np.array(train_labels, dtype=np.int64)
  #test_labels = np.array(test_labels, dtype=np.int64)

  train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
  test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

  #assert train_data.min() == 0.
  #assert train_data.max() == 1.
  #assert test_data.min() == 0.
  #assert test_data.max() == 1.

  return train_data, train_labels, test_data, test_labels


# Load training and test data.
train_data, train_labels, test_data, test_labels = load_mnist()

cifar_train = train_data, train_labels
cifar_test = test_data, test_labels

CLIENTS = 3
SIZE = 10000

def get_data(source):

    all_data = (np.array(source[0][:SIZE*CLIENTS]), source[1][:SIZE*CLIENTS])

    split_data = []
    for s in range(CLIENTS):
        start = s*SIZE
        end = s*SIZE + SIZE
        split_data.append((all_data[0][start:end], all_data[1][start:end]))

    external_data = (np.array(source[0][SIZE*CLIENTS:]), source[1][SIZE*CLIENTS:])

    return all_data, split_data, external_data

CLIENTS = 3
SIZE = 1000

def get_test_data(source):

    all_data = (np.array(source[0][:SIZE*CLIENTS]), source[1][:SIZE*CLIENTS])

    split_data = []
    for s in range(CLIENTS):
        start = s*SIZE
        end = s*SIZE + SIZE
        split_data.append((all_data[0][start:end], all_data[1][start:end]))

    external_data = (np.array(source[0][SIZE*CLIENTS:]), source[1][SIZE*CLIENTS:])

    return all_data, split_data, external_data

cifar_train_data, cifar_train_fed_data, attacker_data = get_data(cifar_train)
cifar_test_data, cifar_test_fed_data, externat_test_data = get_test_data(cifar_test)


class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

class MCAlphaDropout(tf.keras.layers.AlphaDropout):
    def call(self, inputs):
        return super().call(inputs, training=True)
 # this model is centralized    
def create_compiled_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=[28, 28, 1]),
        #tf.keras.layers.AlphaDropout(rate=0.2),
        tf.keras.layers.Dropout(0.2),
        #tf.keras.layers.GaussianNoise(0.2),
        #tf.keras.layers.GaussianDropout(0.2),
        tf.keras.layers.Dense(300, activation="relu"),
        #tf.keras.layers.AlphaDropout(rate=0.2),
        tf.keras.layers.Dropout(0.2),
        #tf.keras.layers.GaussianNoise(0.2),
        #tf.keras.layers.GaussianDropout(0.2),
        tf.keras.layers.Dense(100, activation="relu"),
        #tf.keras.layers.AlphaDropout(rate=0.2),
        tf.keras.layers.Dropout(0.2),
        #tf.keras.layers.GaussianNoise(0.2),
        #tf.keras.layers.GaussianDropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    #mc_model = tf.keras.models.Sequential([
        #MCAlphaDropout(layer.rate) if isinstance(layer, tf.keras.layers.AlphaDropout) else layer
        #for layer in model.layers
    #])
    #mc_model = tf.keras.models.Sequential([
        #MCDropout(layer.rate) if isinstance(layer, tf.keras.layers.Dropout) else layer
        #for layer in model.layers
    #])
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(loss=loss,optimizer = tf.keras.optimizers.SGD(learning_rate=0.01), metrics=["accuracy"])
    #model.compile(loss=loss,optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01), metrics=["accuracy"])
    #model.compile(loss=loss,optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01),metrics=["accuracy"])
    #model.compile(loss=loss,optimizer = tf.keras.optimizers.Ftrl(learning_rate=0.01),metrics=["accuracy"])
    #model.compile(loss=loss,optimizer = tf.keras.optimizers.ada(learning_rate=0.01),metrics=["accuracy"])

    #return mc_model
    return model

centralized_model = create_compiled_keras_model()
centralized_model.summary()

history_callback = centralized_model.fit(cifar_train_data[0], cifar_train_data[1], validation_data=cifar_test_data, batch_size=32, epochs=12, verbose=1)  

print("Cifar, Centralized, IDD, minibatch_size: 32")
print("NFTrain = {}".format(history_callback.history["loss"]))
print("NFTest = {}".format(history_callback.history["val_loss"]))
print("NFAccuracy = {}".format(history_callback.history["val_accuracy"]))

single_model0 = create_compiled_keras_model()
history_callback0 = single_model0.fit(cifar_train_fed_data[0][0], cifar_train_fed_data[0][1], validation_data=(cifar_test_fed_data[0][0],cifar_test_fed_data[0][1]), batch_size=32, epochs=12, verbose=1)

print("Cifar, Ind0, IDD, minibatch_size: 32")
print("NFTrain = {}".format(history_callback0.history["loss"]))
print("NFTest = {}".format(history_callback0.history["val_loss"]))
print("NFAccuracy = {}".format(history_callback0.history["val_accuracy"]))

single_model1 = create_compiled_keras_model()
history_callback1 = single_model1.fit(cifar_train_fed_data[1][0], cifar_train_fed_data[1][1], validation_data=(cifar_test_fed_data[1][0],cifar_test_fed_data[1][1]), batch_size=32, epochs=12, verbose=1)

print("Cifar, Ind1, IDD, minibatch_size: 32")
print("NFTrain = {}".format(history_callback1.history["loss"]))
print("NFTest = {}".format(history_callback1.history["val_loss"]))
print("NFAccuracy = {}".format(history_callback1.history["val_accuracy"]))

single_model2 = create_compiled_keras_model()
history_callback2 = single_model2.fit(cifar_train_fed_data[2][0], cifar_train_fed_data[2][1], validation_data=(cifar_test_fed_data[2][0],cifar_test_fed_data[2][1]), batch_size=32, epochs=12, verbose=1)

print("Cifar, Ind2, IDD, minibatch_size: 32")
print("NFTrain = {}".format(history_callback2.history["loss"]))
print("NFTest = {}".format(history_callback2.history["val_loss"]))
print("NFAccuracy = {}".format(history_callback2.history["val_accuracy"]))

SingAvTrain = (np.array(history_callback0.history["loss"])
 + np.array(history_callback1.history["loss"])
 + np.array(history_callback2.history["loss"])) / 3
SingAvTest = (np.array(history_callback0.history["val_loss"])
 + np.array(history_callback1.history["val_loss"])
 + np.array(history_callback2.history["val_loss"])) / 3
SingAvAcc = (np.array(history_callback0.history["val_accuracy"])
 + np.array(history_callback1.history["val_accuracy"])
 + np.array(history_callback2.history["val_accuracy"])) / 3
print("Cifar10, SingleAverage, IDD, minibatch_size: 32")
print("SingAvTrain = {}".format(SingAvTrain))
print("SingAvTest = {}".format(SingAvTest))
print("SingAvAcc = {}".format(SingAvAcc))
probabilities0 = single_model0.predict(cifar_test_data[0], batch_size=32, verbose=1)
probabilities1 = single_model1.predict(cifar_test_data[0], batch_size=32, verbose=1)
probabilities2 = single_model2.predict(cifar_test_data[0], batch_size=32, verbose=1)
probs = (probabilities0 + probabilities1 + probabilities2) / 3
val_loss = tf.keras.losses.categorical_crossentropy(cifar_test_data[1], probs, from_logits=False)
print(np.mean(val_loss.numpy()))


from sklearn.metrics import accuracy_score

y_pred = probs.argmax(axis=1)
y_true = cifar_test_data[1].argmax(axis=1)

accuracy_score(y_true, y_pred)

cyc_transfer_model = create_compiled_keras_model()
CycTrTrain = []
CycTrTest = []
CycTrAcc = []

for r in range(6):
    for c in range(CLIENTS):
        cyc_history_callback = cyc_transfer_model.fit(cifar_train_fed_data[c][0], cifar_train_fed_data[c][1],
                                                     validation_data=cifar_test_data, batch_size=250, epochs=2)
        CycTrTrain.append(cyc_history_callback.history['loss'])
        CycTrTest.append(cyc_history_callback.history['val_loss'])
        CycTrAcc.append(cyc_history_callback.history['val_accuracy'])
        
        
print("Cifar10, Cyclic Weight Transfer, IDD, minibatch_size: 32")
print("CycTrTrain = {}".format(CycTrTrain))
print("CyctTrTest = {}".format(CycTrTest))
print("CycTrAccuracy = {}".format(CycTrAcc))

initial_model = create_compiled_keras_model()

FedTrain = []
FedTest = []
FedAcc = []

for r in range(18):

    deltas = []

    for c in range(CLIENTS):

        federated_model = create_compiled_keras_model()

        federated_model.set_weights(initial_model.get_weights())

        #fed_history_callback = federated_model.fit(cifar_train_fed_data[c][0], cifar_train_fed_data[c][1],
         #                                      batch_size=250, epochs=10, verbose=1)
        fed_history_callback = federated_model.fit(cifar_train_fed_data[c][0], cifar_train_fed_data[c][1],
                                               batch_size=32, epochs=10, verbose=1)

        delta = np.array(initial_model.get_weights()) - np.array(federated_model.get_weights())

        deltas.append(delta)

    print('Epoch {}/18'.format(r+1))
    delt_av = (deltas[0] + deltas[1] + deltas[2]) / 3
    new_weights = np.array(initial_model.get_weights()) - delt_av
    initial_model.set_weights(new_weights)

    FedTrain.append(initial_model.evaluate(cifar_train_data[0], cifar_train_data[1])[0])
    validation = initial_model.evaluate(cifar_test_data[0], cifar_test_data[1])
    FedTest.append(validation[0])
    FedAcc.append(validation[1])
    
print("Cifar10, Federated Model, IDD, minibatch_size: 32")
print("FedTrainE1B32 = {}".format(FedTrain))
print("FedTestE1B32 = {}".format(FedTest))
print("FedAccuracyE1B32 = {}".format(FedAcc))








############################################# This part starts attacking to the Federated and centralized model
import numpy as np

from absl import app
from absl import flags

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data

NUM_CLASSES = 10

SHADOW_DATASET_SIZE = 1000
ATTACK_TEST_DATASET_SIZE = 5000

num_shadows = 10

# This is for federated models
def target_model_fn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=[28, 28, 1]),
        tf.keras.layers.Dropout(0.2),
        #tf.keras.layers.AlphaDropout(rate=0.2),
        #tf.keras.layers.GaussianNoise(0.2),
        #tf.keras.layers.GaussianDropout(0.2),
        tf.keras.layers.Dense(300, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        #tf.keras.layers.AlphaDropout(rate=0.2),
        #tf.keras.layers.GaussianNoise(0.2),
        #tf.keras.layers.GaussianDropout(0.2),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        #tf.keras.layers.AlphaDropout(rate=0.2),
        #tf.keras.layers.GaussianNoise(0.2),
        #tf.keras.layers.GaussianDropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    #mc_model = tf.keras.models.Sequential([
        #MCAlphaDropout(layer.rate) if isinstance(layer, tf.keras.layers.AlphaDropout) else layer
        #for layer in model.layers
    #])
    #mc_model = tf.keras.models.Sequential([
        #MCDropout(layer.rate) if isinstance(layer, tf.keras.layers.Dropout) else layer
        #for layer in model.layers
    #])
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    #model.compile(loss=loss,      optimizer = tf.keras.optimizers.SGD(learning_rate=0.01),     metrics=["accuracy"])
    #model.compile(loss=loss,optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01), metrics=["accuracy"])
    model.compile(loss=loss,optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01),metrics=["accuracy"])
    
    #model.compile(loss=loss,optimizer = tf.keras.optimizers.Ftrl(learning_rate=0.01),metrics=["accuracy"])
    #model.compile(loss=loss,optimizer = tf.keras.optimizers.Adafactor(learning_rate=0.01),metrics=["accuracy"])
    #return mc_model
    return model

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

#target_model = federated_model
target_model = single_model0

# Train the shadow models.
smb = ShadowModelBundle(
    target_model_fn,
    shadow_dataset_size=SHADOW_DATASET_SIZE,
    num_models=num_shadows
)

# Using cifar10 test set to train shadow models
attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
    cifar_test[0], cifar_test[1], test_size=0.5)

print(attacker_X_train.shape, attacker_X_test.shape)

print("Training the shadow models...")
X_shadow, y_shadow = smb.fit_transform(
    attacker_X_train,
    attacker_y_train,
    fit_kwargs=dict(
        epochs=32, #should be 32
        verbose=True,
        validation_data=(attacker_X_test, attacker_y_test)
    )
)

# ShadowModelBundle returns data in the format suitable for the AttackModelBundle.
amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)

# Fit the attack models.
print("Training the attack models...")
amb.fit(X_shadow, y_shadow, fit_kwargs=dict(epochs=32, verbose=True))

target_data = cifar_train_fed_data[0]
attacker_data = cifar_test_fed_data[0]

attack_test_data, real_membership_labels = prepare_attack_data(target_model, target_data, attacker_data)

def results(attack_guesses,real_membership_labels):
    pred_labels=attack_guesses
    true_labels=real_membership_labels

    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

    print ('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))

    print("acc = " + str((TP+TN)/(TP+TN+FP+FN)))
    print("precision = " + str((TP)/(TP+FP)))
    print("recall = " + str((TP)/(TP+FN)))

    acc= (TP+TN)/(TP+TN+FP+FN)
    prec=(TP)/(TP+FP)
    rec=(TP)/(TP+FN)
    return [acc,prec,rec]

attack_guesses = amb.predict(attack_test_data)
attack_precision = np.mean((attack_guesses == 1) == (real_membership_labels == 1))

class_precision = []

for c in range(NUM_CLASSES):
    #attack_test_data, real_membership_labels = prepare_attack_data(centralized_model, cifar_train_data, attacker_data)
    target_indices = [i for i, d in enumerate(target_data[1].argmax(axis=1)) if d == c]
    test_indices = [i for i, d in enumerate(attacker_data[1].argmax(axis=1)) if d == c]


    print(np.sum(attack_guesses[target_indices]==1) / (np.sum(attack_guesses[target_indices]) + np.sum(attack_guesses[SIZE:][test_indices])))

    class_precision.append(
            np.sum(attack_guesses[target_indices]==1) / (np.sum(attack_guesses[target_indices])
                                                     + np.sum(attack_guesses[SIZE:][test_indices])))
print("Average Accuracy: ", attack_precision)

result=results(attack_guesses,real_membership_labels)

    #attack_accuracy_class[c].append(result)
    
import pickle
import argparse

import numpy as np
import torch
from numpy import linalg as LA
from sklearn.metrics import precision_score, recall_score
from sklearn.cluster import SpectralClustering

np.random.seed(seed=14)
torch.manual_seed(14)

target_model = federated_model

parser = argparse.ArgumentParser()
parser.add_argument('--n_sample', type=int, default=5000)
parser.add_argument('--n_attack', type=int, default=50)
parser.add_argument('--seed', type=int, default=140)
parser.add_argument('--neighbors', type=int, default=40)
parser.add_argument('--data_generate', type=bool, default=False)
attack_args = parser.parse_args(args=[])

precisions = []
recalls = []
f1_scores = []

np.random.seed(seed=attack_args.seed)
torch.manual_seed(attack_args.seed)

import numpy as np
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from collections import Counter
from sklearn.model_selection import train_test_split



def data_reader(data_name = "mnist"):
    file_path = "data/"
    #data = pd.read_csv(file_path + 'mnist_train.csv', header=1)
    #data = pd.read_csv(file_path + 'fashion-mnist_train.csv', header=1)
    data = pd.read_csv(file_path + 'fashion-mnist_train.csv', header=1, skiprows=30000, nrows=29999)
    #data = pd.read_csv(file_path + 'fashion-mnist_test.csv', header=0)
    data = np.array(data)
    labels = data[:,0]
    data = data[:,1:]

    categorical_features = []

    data = data/data.max()
    oh_encoder = ColumnTransformer(
    [('oh_enc', OneHotEncoder(sparse=False), categorical_features),],
    remainder='passthrough' )
    oh_data = oh_encoder.fit_transform(data)

    #randomly select 10000 records as training data
    train_idx = np.random.choice(len(labels), 9999, replace = False)
    idx = range(len(labels))
    idx = np.array(idx)
    test_idx = list(set(idx).difference(set(train_idx)))
    test_idx = np.array(test_idx)

    assert test_idx.sum() + train_idx.sum() == idx.sum()

    X_train = data[train_idx,:]
    Y_train = labels[train_idx]

    X_test = data[test_idx,:]
    Y_test = labels[test_idx]

    orig_dataset = {"X_train":X_train,
               "Y_train":Y_train,
               "X_test":X_test,
               "Y_test":Y_test}

    X_train = oh_data[train_idx,:]

    X_test = oh_data[test_idx,:]

    oh_dataset = {"X_train":X_train,
               "Y_train":Y_train,
               "X_test":X_test,
               "Y_test":Y_test}

    return orig_dataset, oh_dataset, oh_encoder

orig_dataset, oh_dataset, OH_Encoder = data_reader("mnist")
class_label_for_count = np.unique(np.hstack([orig_dataset["Y_train"], orig_dataset["Y_test"]]))
n_class = len(class_label_for_count)
n_features = orig_dataset['X_train'].shape[1]
y_attack = np.hstack(([np.ones(int(attack_args.n_attack/2)), np.zeros(int(attack_args.n_attack/2))]))
x_attack = np.zeros((int(attack_args.n_attack), n_features))
Jacobian_matrix = np.zeros([attack_args.n_attack, n_class, n_features])
if attack_args.data_generate:
    output_x = np.zeros((attack_args.n_attack, n_features))
    output_y = y_attack
    classes = np.zeros((attack_args.n_attack, 1))
    
def fn_R_given_Selected(dataset, IN_or_OUT = 1):
    if(IN_or_OUT == 1):#IN_or_OUT == 1 meaning selecting R_given from training set
        idx = np.random.choice( len(dataset['Y_train']) )
        R_given = dataset['X_train'][idx,:]
        R_given_y = dataset['Y_train'][idx]
    elif(IN_or_OUT == 0):#IN_or_OUT == 0 meaning selecting R_given from testing set
        idx = np.random.choice( len(dataset['Y_test']) )
        R_given = dataset['X_test'][idx,:]
        R_given_y = dataset['Y_test'][idx]
    return R_given, R_given_y    


def Target_Model_pred_fn(Target_Model, X_test):
    #pred_proba = Target_Model.predict(X_test)
    predictions = Target_Model.predict(X_test)
    pred_proba = np.exp(predictions) / np.sum(np.exp(predictions), axis=1,keepdims=True)
    return pred_proba

categorical_list ={
    "mnist": [1,2,3,4,5,6,7,8,9,10],
}

def fn_Sample_Generator(R_given, dataset):
    if not dataset in categorical_list.keys():
        dataset = "null"
    epsilon = 1e-6
    R_given = R_given.reshape([1, -1])
    n_feature = R_given.shape[1]
    local_samples = np.repeat(R_given, repeats=n_feature, axis=0)
    for i in range(n_feature):
        if i in categorical_list[dataset]:
            continue
        local_samples[i][i] += epsilon

    return local_samples

def fn_Jacobian_Calculation(R_given, local_proba, n_features, n_class):
    epsilon = 1e-6
    jacobian = np.zeros([n_class, n_features])

    for ii in range(n_class):
        jacobian[ii, :] = (local_proba[:, ii] - R_given[ii]) / epsilon
    return jacobian

for ii in range(attack_args.n_attack):
  R_x, R_y = fn_R_given_Selected(orig_dataset, IN_or_OUT=y_attack[ii])
  R_x_OH = OH_Encoder.transform(R_x.reshape(1, -1))
  x_attack[ii] = R_x
  local_samples = fn_Sample_Generator(R_x, "mnist")
  oh_local_samples = OH_Encoder.transform(local_samples)
  local_proba = Target_Model_pred_fn(target_model, oh_local_samples)
  R_local_proba = Target_Model_pred_fn(target_model, R_x_OH)
  Jacobian_matrix[ii] = fn_Jacobian_Calculation(R_local_proba[0], local_proba, n_features, n_class)

  if attack_args.data_generate:
      output_x[ii] = R_x
      classes[ii] = R_y
    
    
Jacobian_norms = LA.norm(Jacobian_matrix, axis=(1, 2))

split = 1
attack_cluster = SpectralClustering(n_clusters=6, n_jobs=-1, affinity='nearest_neighbors', n_neighbors=19)
y_attack_pred = attack_cluster.fit_predict(Jacobian_norms.reshape(-1, 1))
cluster_1 = np.where(y_attack_pred >= split)[0]
cluster_0 = np.where(y_attack_pred < split)[0]
y_attack_pred[cluster_1] = 1
y_attack_pred[cluster_0] = 0
cluster_1_mean_norm = Jacobian_norms[cluster_1].mean()
cluster_0_mean_norm = Jacobian_norms[cluster_0].mean()
if cluster_1_mean_norm > cluster_0_mean_norm:
  y_attack_pred = np.abs(y_attack_pred-1)
  
precision = precision_score(y_attack, y_attack_pred)
recall = recall_score(y_attack, y_attack_pred)
f1_score = 2*precision*recall/(precision+recall)
print(precision, recall, f1_score)
precisions.append(precision)
recalls.append(recall)
f1_scores.append(f1_score)

  