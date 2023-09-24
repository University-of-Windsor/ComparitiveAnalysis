# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:54:16 2023

@author: aliab
"""

import numpy as np
import tensorflow as tf
import random
import datetime
import tensorflow_privacy
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow_privacy

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

from utils import *
from Distiller import *

global_optimizer='nadam'
global_dataset='fmnist'
global_countermeasure = ['gn']

tf.random.set_seed(42)
np.random.seed(42)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

train_data, train_labels, test_data, test_labels = load_dataset(dataset_name=global_dataset)
cifar_train = train_data, train_labels
cifar_test = test_data, test_labels

cifar_train_data, cifar_train_fed_data, attacker_data = get_data(cifar_train)


cifar_test_data, cifar_test_fed_data, externat_test_data = get_test_data(cifar_test)


    
#tf.keras.layers.Input(shape=(28, 28, 1)),
#tf.keras.layers.Flatten(),
#tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=None),
#tf.keras.layers.LayerNormalization(),


l2_norm_clip = 1.5
noise_multiplier = 1.3
num_microbatches = 8
learning_rate = 0.01



        
    #optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
    #l2_norm_clip=l2_norm_clip,
    #noise_multiplier=noise_multiplier,
    #num_microbatches=num_microbatches,
    #learning_rate=learning_rate)
    #loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
    #model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    #model.summary()
    #return model
    
    
centralized_model = create_compiled_keras_model(optimizer=global_optimizer,dataset_name=global_dataset,countermeasures=global_countermeasure)
centralized_model.summary()

#tf.keras.utils.plot_model(centralized_model, "my_first_model_with_shape_info.png", show_shapes=True)

history_callback = centralized_model.fit(cifar_train_data[0], cifar_train_data[1], validation_data=cifar_test_data, batch_size=32, epochs=12, verbose=1)
print("Cifar, Centralized, IDD, minibatch_size: 32")
#print("NFTrain = {}".format(history_callback.history["loss"]))
#print("NFTest = {}".format(history_callback.history["val_loss"]))
print("NFAccuracy = {}".format(history_callback.history["val_accuracy"]))

np.mean(history_callback.history['val_accuracy'])
#np.mean(history_callback.history['val_loss'])
#np.mean(history_callback.history['loss'])
#np.mean(history_callback.history['accuracy'])


############################ Import dataset and model for FL
def run_experiment_onFL_model(optimizer='sgd', dataset_name='mnist'):
    single_model0 = create_compiled_keras_model(optimizer=optimizer,dataset_name=dataset_name)
    history_callback0 = single_model0.fit(cifar_train_fed_data[0][0], cifar_train_fed_data[0][1], validation_data=(cifar_test_fed_data[0][0],cifar_test_fed_data[0][1]), batch_size=32, epochs=12, verbose=1)

    single_model1 = create_compiled_keras_model(optimizer=optimizer,dataset_name=dataset_name)
    history_callback1 = single_model1.fit(cifar_train_fed_data[1][0], cifar_train_fed_data[1][1], validation_data=(cifar_test_fed_data[1][0],cifar_test_fed_data[1][1]), batch_size=32, epochs=12, verbose=1)


    single_model2 = create_compiled_keras_model(optimizer=optimizer,dataset_name=dataset_name)
    history_callback2 = single_model2.fit(cifar_train_fed_data[2][0], cifar_train_fed_data[2][1], validation_data=(cifar_test_fed_data[2][0],cifar_test_fed_data[2][1]), batch_size=32, epochs=12, verbose=1)


#client 1,2,3 average
   # SingAvTrain = (np.array(history_callback0.history["loss"])
   #                + np.array(history_callback1.history["loss"])
   #                + np.array(history_callback2.history["loss"])) / 3
   # SingAvTest = (np.array(history_callback0.history["val_loss"])
    #              + np.array(history_callback1.history["val_loss"])
   #               + np.array(history_callback2.history["val_loss"])) / 3
   # SingAvAcc = (np.array(history_callback0.history["val_accuracy"])
   #              + np.array(history_callback1.history["val_accuracy"])
   #              + np.array(history_callback2.history["val_accuracy"])) / 3


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

#cyclic weigths transfer
    cyc_transfer_model = create_compiled_keras_model(optimizer=optimizer,dataset_name=dataset_name)
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

#federated learning
    initial_model = cyc_transfer_model#create_compiled_keras_model(optimizer=optimizer,dataset_name=dataset_name)

    FedTrain = []
    FedTest = []
    FedAcc = []

    for r in range(5):

        deltas = []

        for c in range(CLIENTS):

            federated_model = create_compiled_keras_model(optimizer=optimizer,dataset_name=dataset_name)

            federated_model.set_weights(initial_model.get_weights())

        #fed_history_callback = federated_model.fit(cifar_train_fed_data[c][0], cifar_train_fed_data[c][1],
        #                                      batch_size=250, epochs=10, verbose=1)
            fed_history_callback = federated_model.fit(cifar_train_fed_data[c][0], cifar_train_fed_data[c][1], batch_size=32, epochs=10, verbose=1)

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
    
    #federated_model.save("data/mnistfederated.h5")
   
    return FedAcc, FedTest, federated_model

FedAccuracy, FedTest, federated_model = run_experiment_onFL_model(optimizer=global_optimizer,dataset_name=global_dataset)




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
#num_shadows = 1



def attack_model_fn():
    model = tf.keras.models.Sequential()

    model.add(layers.Dense(64, activation="relu", input_shape=(10,)))
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(0.4, noise_shape=None, seed=None))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model





############################################################Start of attack 1 execution
#target_model = federated_model
target_model = centralized_model

# Train the shadow models.
def target_model_fn():
    return create_compiled_keras_model(optimizer=global_optimizer,dataset_name=global_dataset,countermeasures=global_countermeasure)
smb = ShadowModelBundle(
    target_model_fn,
    shadow_dataset_size=SHADOW_DATASET_SIZE,
    num_models=num_shadows
)

# Using cifar10 test set to train shadow models
attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(cifar_test[0], cifar_test[1], test_size=0.5)

#attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
    #attack3_test[0], attack3_test[1], test_size=0.5)

print(attacker_X_train.shape, attacker_X_test.shape)

print("Training the shadow models...")
X_shadow, y_shadow = smb.fit_transform(
    attacker_X_train,
    attacker_y_train,
    fit_kwargs=dict(
        epochs=32,
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

#target_data = cifar_train_data
#attacker_data = attacker_data

#attack_test_data, real_membership_labels = prepare_attack_data(federated_model, cifar_train_data, attacker_data)
attack_test_data, real_membership_labels = prepare_attack_data(target_model, target_data, attacker_data)





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


################################################# End of Attack 1 execution








######################################## Start of attack 4
    #attack_accuracy_class[c].append(result)
    
import pickle
import argparse

import numpy as np
import torch
from numpy import linalg as LA
from sklearn.metrics import precision_score, recall_score
from sklearn.cluster import SpectralClustering, KMeans

np.random.seed(seed=14)
torch.manual_seed(14)

#target_model = federated_model
target_model = centralized_model

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
    if(data_name=='mnist'):
        #data = pd.read_csv(file_path + 'mnist_train.csv', header=1)
        data = pd.read_csv(file_path + 'mnist_train.csv', header=1, skiprows=30000, nrows=29999)
    if(data_name=='fmnist'):
        #data = pd.read_csv(file_path + 'fashion-mnist_train.csv', header=1)
        data = pd.read_csv(file_path + 'fashion-mnist_train.csv', header=1, skiprows=30000, nrows=29999)
        #data = pd.read_csv(file_path + 'fashion-mnist_test.csv', header=0)
    if(data_name=='cifar10'):
        data = pd.read_csv(file_path + 'cifar_train.csv', header=1, skiprows=30000, nrows=29999)
        
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

# if global_dataset=='fmnist' or global_dataset=='cifar10':
#     categorical_list ={"fmnist":[0,1,2,3,4,5,6,7,8,9]}
# else:
categorical_list ={"mnist": [0,1,2,3,4,5,6,7,8,9],"fmnist": [0,1,2,3,4,5,6,7,8,9]}
orig_dataset, oh_dataset, OH_Encoder = data_reader(global_dataset)

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
    






for ii in range(attack_args.n_attack):
      R_x, R_y = fn_R_given_Selected(orig_dataset, IN_or_OUT=y_attack[ii])
      R_x_OH = OH_Encoder.transform(R_x.reshape(1, -1))
      x_attack[ii] = R_x
      local_samples = fn_Sample_Generator(R_x, global_dataset)
      oh_local_samples = OH_Encoder.transform(local_samples)
      samples_reshape=oh_local_samples.reshape((-1,28,28,1))
      local_proba = Target_Model_pred_fn(target_model, oh_local_samples)
      #local_proba = Target_Model_pred_fn(target_model, samples_reshape)

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

print("average")
print(sum(precisions)/len(precisions), sum(recalls)/len(recalls), sum(f1_scores)/len(f1_scores))

######################################## End of Attack 4







#ML-leaks

import sys
from sklearn.metrics import classification_report, accuracy_score
import theano.tensor as T
import numpy as np
import lasagne
import theano
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
sys.dont_write_bytecode = True

def iterate_minibatches(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    start_idx = None
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

    if start_idx is not None and start_idx + batch_size < len(inputs):
        excerpt = indices[start_idx + batch_size:] if shuffle else slice(start_idx + batch_size, len(inputs))
        yield inputs[excerpt], targets[excerpt]
        
def get_cnn_model(n_in, n_hidden, n_out):
    net = dict()
    net['input'] = lasagne.layers.InputLayer(shape=(None, n_in[1], n_in[2], n_in[3]))

    net['conv1'] = lasagne.layers.Conv2DLayer(net['input'], num_filters=32, filter_size=(5, 5),
            pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))

    #net['maxPool1'] = lasagne.layers.MaxPool2DLayer(net['conv1'], pool_size=(2, 2))

    net['conv2'] = lasagne.layers.Conv2DLayer(
            net['maxPool1'], num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
    #net['maxPool2'] = lasagne.layers.MaxPool2DLayer(net['conv2'], pool_size=(2, 2))

    net['fc'] = lasagne.layers.DenseLayer(
        net['maxPool2'],
        num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.tanh)

    net['output'] = lasagne.layers.DenseLayer(
        net['fc'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net

def get_nn_model(n_in, n_hidden, n_out):
    net = dict()
    print(n_in)
    net['input'] = lasagne.layers.InputLayer((None, n_in[1]))
    net['fc'] = lasagne.layers.DenseLayer(
        net['input'],
        num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.tanh)
    net['output'] = lasagne.layers.DenseLayer(
        net['fc'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net

def train_model(dataset, n_hidden=50, batch_size=32, epochs=12, learning_rate=0.01, model='cnn', l2_ratio=1e-7):


    train_x, train_y, test_x, test_y = dataset
    n_in = train_x.shape

    n_out = len(np.unique(train_y))

    if batch_size > len(train_y):
        batch_size = len(train_y)
    print('Building model with {} training data, {} classes...'.format(len(train_x), n_out))
    if model=='cnn' or model=='cnn2' or model=='Droppcnn' or  model=='Droppcnn2':
        input_var = T.tensor4('x')
    else:
        input_var = T.matrix('x')
    target_var = T.ivector('y')
    if model == 'cnn':
        print('Using a multilayer convolution neural network based model...')
        net = get_cnn_model(n_in, n_hidden, n_out)
    elif model == 'nn':
        print('Using a multilayer neural network based model...')
        net = get_nn_model(n_in, n_hidden, n_out)
    #else:
        #print('Using a single layer softmax based model...')
        #net = get_softmax_model(n_in, n_out)

    net['input'].input_var = input_var

    output_layer = net['output']
    # create loss function
    prediction = lasagne.layers.get_output(output_layer)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean() + l2_ratio * lasagne.regularization.regularize_network_params(output_layer,
                                                                                 lasagne.regularization.l2)
    # create parameter update expressions
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    # use trained network for predictions
    test_prediction = lasagne.layers.get_output(output_layer, deterministic=True)
    test_fn = theano.function([input_var], test_prediction)
    print('Training...')
    counter = 1
    for epoch in range(epochs):
        loss = 0
        for input_batch, target_batch in iterate_minibatches(train_x, train_y, batch_size):
            loss += train_fn(input_batch, target_batch)

        loss = round(loss, 3)
        if(epoch % 10 ==0):
            print('Epoch {}, train loss {}'.format(epoch, loss))


        counter = counter +1
    pred_y = []
    for input_batch, _ in iterate_minibatches(train_x, train_y, batch_size, shuffle=False):
        #input_batch = (np.reshape(input_batch,(len(input_batch),3,32,32)))
        pred = test_fn(input_batch)
        pred_y.append(np.argmax(pred, axis=1))
    pred_y = np.concatenate(pred_y)

    if test_x is not None:
        print('Testing...')
        pred_y = []

        if batch_size > len(test_y):
            batch_size = len(test_y)

        for input_batch, _ in iterate_minibatches(test_x, test_y, batch_size, shuffle=False):
            #input_batch = (np.reshape(input_batch,(len(input_batch),3,32,32)))
            pred = test_fn(input_batch)
            pred_y.append(np.argmax(pred, axis=1))
        pred_y = np.concatenate(pred_y)
        print('Testing Accuracy: {}'.format(accuracy_score(test_y, pred_y)))


    print('More detailed results:')
    print(classification_report(test_y, pred_y))


    return output_layer


#Deep Learning

import sys

sys.dont_write_bytecode = True

#from classifier import train_model, iterate_minibatches
import numpy as np
import theano.tensor as T
import lasagne
import theano

np.random.seed(21312)

def train_target_model(dataset,epochs=12, batch_size=32, learning_rate=0.01, l2_ratio=1e-7,
                       n_hidden=50, model='nn'):


    train_x, train_y, test_x, test_y = dataset

    #output_layer = train_model(dataset, n_hidden=n_hidden, epochs=epochs, learning_rate=learning_rate,
                               #batch_size=batch_size, model=model, l2_ratio=l2_ratio)
    output_layer = federated_model
    # test data for attack model
    attack_x, attack_y = [], []
    if model=='cnn':
        #Dimension for CIFAR-10
        input_var = T.tensor4('x')
    else:
        #Dimension for News
        input_var = T.matrix('x')

    prob = lasagne.layers.get_output(output_layer, input_var, deterministic=True)

    prob_fn = theano.function([input_var], prob)

    # data used in training, label is 1
    for batch in iterate_minibatches(train_x, train_y, batch_size, False):
        attack_x.append(prob_fn(batch[0]))
        attack_y.append(np.ones(len(batch[0])))

    # data not used in training, label is 0
    for batch in iterate_minibatches(test_x, test_y, batch_size, False):
        attack_x.append(prob_fn(batch[0]))
        attack_y.append(np.zeros(len(batch[0])))

    #print len(attack_y)
    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    #print('total length  ' + str(sum(attack_y)))
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')

    return attack_x, attack_y, output_layer


#ML-Leaks

import sys
sys.dont_write_bytecode = True

import numpy as np

import pickle
from sklearn.model_selection import train_test_split
import random
import lasagne
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import argparse
#import deeplearning as dp
#import classifier


parser = argparse.ArgumentParser()
parser.add_argument('--adv',  default='1', help='Which adversary 1, 2, or 3')
parser.add_argument('--dataset', default='CIFAR10', help='Which dataset to use (CIFAR10 or News)')
parser.add_argument('--classifierType', default='cnn', help='Which classifier cnn or nn')
parser.add_argument('--dataset2', default='Mnist', help='Which second dataset for adversary 2 (CIFAR10 or MNIST)')
parser.add_argument('--classifierType2', default='nn', help='Which classifier cnn or nn')
parser.add_argument('--dataFolderPath', default='/content/drive/My Drive/Proposal work/MLeaks/data/', help='Path to store data')
parser.add_argument('--pathToLoadData', default='/content/drive/My Drive/Proposal work/MLeaks/data/cifar-10', help='Path to load dataset from')
#parser.add_argument('--pathToLoadData', default='/content/drive/My Drive/Proposal work/MLeaks/data/mnist', help='Path to load dataset from')
parser.add_argument('--num_epoch', type=int, default=12, help='Number of epochs to train shadow/target models')
parser.add_argument('--preprocessData', action='store_true', help='Preprocess the data, if false then load preprocessed data')
parser.add_argument('--trainTargetModel', action='store_true', help='Train a target model, if false then load an already trained model')
parser.add_argument('--trainShadowModel', action='store_true', help='Train a shadow model, if false then load an already trained model')

opt = parser.parse_args([])

def clipDataTopX(dataToClip, top=3):
	res = [ sorted(s, reverse=True)[0:top] for s in dataToClip ]
	return np.array(res)

'''def readCIFAR10(data_path):
	for i in range(5):
		f = open(data_path + '/data_batch_' + str(i + 1), 'rb')
		train_data_dict = pickle.load(f, encoding='latin1')
		f.close()
		if i == 0:
			X = train_data_dict["data"]
			y = train_data_dict["labels"]
			continue
		X = np.concatenate((X , train_data_dict["data"]),   axis=0)
		y = np.concatenate((y , train_data_dict["labels"]), axis=0)
	f = open(data_path + '/test_batch', 'rb')
	test_data_dict = pickle.load(f, encoding='latin1')
	f.close()
	XTest = np.array(test_data_dict["data"])
	yTest = np.array(test_data_dict["labels"])
	return X, y, XTest, yTest
'''

def readCIFAR10():
  train_data_dict, test_data_dict = tf.keras.datasets.cifar10.load_data()

  X = np.array(train_data_dict)
  y = np.array(train_data_dict)
  XTest = np.array(test_data_dict)
  yTest = np.array(test_data_dict)

  return X, y, XTest, yTest

def readMNIST():
  train_data_dict, test_data_dict = tf.keras.datasets.mnist.load_data()

  X = np.array(train_data_dict["data"])
  y = np.array(train_data_dict["labels"])
  XTest = np.array(test_data_dict["data"])
  yTest = np.array(test_data_dict["labels"])

  return X, y, XTest, yTest

def trainTarget(modelType, X, y,
				X_test=[], y_test =[],
				splitData=True,
				test_size=0.5,
				inepochs=50, batch_size=300,
				learning_rate=0.001):


	if(splitData):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
	else:
		X_train = X
		y_train = y

	dataset = (X_train.astype(np.float32),
			   y_train.astype(np.int32),
			   X_test.astype(np.float32),
			   y_test.astype(np.int32))

	attack_x, attack_y, theModel = train_target_model(dataset=dataset, epochs=inepochs, batch_size=batch_size,learning_rate=learning_rate, n_hidden=128,l2_ratio = 1e-07,model=modelType)

	return attack_x, attack_y, theModel

def load_data(data_name):
	with np.load( data_name) as f:
		train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
	return train_x, train_y

def trainAttackModel(X_train, y_train, X_test, y_test):
	dataset = (X_train.astype(np.float32),
			   y_train.astype(np.int32),
			   X_test.astype(np.float32),
			   y_test.astype(np.int32))

	output = train_model(dataset=dataset,
					model='softmax')

	return output


def preprocessingCIFAR(toTrainData, toTestData):
	def reshape_for_save(raw_data):
		raw_data = np.dstack((raw_data[:, :1024], raw_data[:, 1024:2048], raw_data[:, 2048:]))
		raw_data = raw_data.reshape((raw_data.shape[0], 32, 32, 3)).transpose(0,3,1,2)
		return raw_data.astype(np.float32)

	offset = np.mean(reshape_for_save(toTrainData), 0)
	scale  = np.std(reshape_for_save(toTrainData), 0).clip(min=1)

	def rescale(raw_data):
		return (reshape_for_save(raw_data) - offset) / scale

	return rescale(toTrainData), rescale(toTestData)

def preprocessingMnist(toTrainData, toTestData):
	def reshape_for_save(raw_data):
		raw_data = np.dstack((raw_data[:, :1024], raw_data[:, 1024:2048], raw_data[:, 2048:]))
		raw_data = raw_data.reshape((raw_data.shape[0], 32, 32, 3)).transpose(0,3,1,2)
		return raw_data.astype(np.float32)

	offset = np.mean(reshape_for_save(toTrainData), 0)
	scale  = np.std(reshape_for_save(toTrainData), 0).clip(min=1)

	def rescale(raw_data):
		return (reshape_for_save(raw_data) - offset) / scale

	return rescale(toTrainData), rescale(toTestData)

def shuffleAndSplitData(dataX, dataY,cluster):
	c = list(zip(dataX, dataY))
	random.shuffle(c)
	dataX, dataY = zip(*c)
	toTrainData  = np.array(dataX[:cluster])
	toTrainLabel = np.array(dataY[:cluster])

	shadowData  = np.array(dataX[cluster:cluster*2])
	shadowLabel = np.array(dataY[cluster:cluster*2])

	toTestData  = np.array(dataX[cluster*2:cluster*3])
	toTestLabel = np.array(dataY[cluster*2:cluster*3])

	shadowTestData  = np.array(dataX[cluster*3:cluster*4])
	shadowTestLabel = np.array(dataY[cluster*3:cluster*4])

	return toTrainData, toTrainLabel,shadowData,shadowLabel,toTestData,toTestLabel,shadowTestData,shadowTestLabel

def initializeData(dataset,orginialDatasetPath,dataFolderPath = '/content/drive/My Drive/Proposal work/MLeaks/data/'):
	if(dataset == 'CIFAR10'):
		print("Loading data")
		dataX, dataY, _, _ = readCIFAR10()
		print("Preprocessing data")
		cluster = 10520
		dataPath = dataFolderPath+dataset+'/Preprocessed'
		toTrainData, toTrainLabel,shadowData,shadowLabel,toTestData,toTestLabel,shadowTestData,shadowTestLabel = shuffleAndSplitData(dataX, dataY,cluster)
		toTrainDataSave, toTestDataSave    = preprocessingCIFAR(toTrainData, toTestData)
		shadowDataSave, shadowTestDataSave = preprocessingCIFAR(shadowData, shadowTestData)
	elif(dataset == 'MNIST'):
		print("Loading data")
		dataX, dataY, _, _ = readMNIST(orginialDatasetPath)
		print("Preprocessing data")
		cluster = 10520
		dataPath = dataFolderPath+dataset+'/Preprocessed'
		toTrainData, toTrainLabel,shadowData,shadowLabel,toTestData,toTestLabel,shadowTestData,shadowTestLabel = shuffleAndSplitData(dataX, dataY,cluster)
		toTrainDataSave, toTestDataSave    = preprocessingMnist(toTrainData, toTestData)
		shadowDataSave, shadowTestDataSave = preprocessingMnist(shadowData, shadowTestData)


	try:
		os.makedirs(dataPath)
	except OSError:
		pass

	np.savez(dataPath + '/targetTrain.npz', toTrainDataSave, toTrainLabel)
	np.savez(dataPath + '/targetTest.npz',  toTestDataSave, toTestLabel)
	np.savez(dataPath + '/shadowTrain.npz', shadowDataSave, shadowLabel)
	np.savez(dataPath + '/shadowTest.npz',  shadowTestDataSave, shadowTestLabel)

	print("Preprocessing finished\n\n")
    
def initializeTargetModel(dataset,num_epoch,dataFolderPath= '/content/drive/My Drive/Proposal work/MLeaks/data/',modelFolderPath = '/content/drive/My Drive/Proposal work/MLeaks/model/',classifierType = 'cnn'):
	dataPath = dataFolderPath+dataset+'/Preprocessed'
	attackerModelDataPath = dataFolderPath+dataset+'/attackerModelData'
	modelPath = modelFolderPath + dataset
	try:
		os.makedirs(attackerModelDataPath)
		os.makedirs(modelPath)
	except OSError:
		pass
	print("Training the Target model for {} epoch".format(num_epoch))
	targetTrain, targetTrainLabel  = load_data(dataPath + '/targetTrain.npz')
	targetTest,  targetTestLabel   = load_data(dataPath + '/targetTest.npz')
	attackModelDataTarget, attackModelLabelsTarget, targetModelToStore = trainTarget(classifierType,targetTrain, targetTrainLabel, X_test=targetTest, y_test=targetTestLabel, splitData= False, inepochs=num_epoch, batch_size=100)
	np.savez(attackerModelDataPath + '/targetModelData.npz', attackModelDataTarget, attackModelLabelsTarget)
	np.savez(modelPath + '/targetModel.npz', *lasagne.layers.get_all_param_values(targetModelToStore))
	return attackModelDataTarget, attackModelLabelsTarget

def initializeShadowModel(dataset,num_epoch,dataFolderPath= '/content/drive/My Drive/Proposal work/MLeaks/data/',modelFolderPath = '/content/drive/My Drive/Proposal work/MLeaks/model/',classifierType = 'cnn'):
	dataPath = dataFolderPath+dataset+'/Preprocessed'
	attackerModelDataPath = dataFolderPath+dataset+'/attackerModelData'
	modelPath = modelFolderPath + dataset
	try:
		os.makedirs(modelPath)
	except OSError:
		pass
	print("Training the Shadow model for {} epoch".format(num_epoch))
	shadowTrainRaw, shadowTrainLabel  = load_data(dataPath + '/shadowTrain.npz')
	targetTestRaw,  shadowTestLabel   = load_data(dataPath + '/shadowTest.npz')
	attackModelDataShadow, attackModelLabelsShadow, shadowModelToStore = trainTarget(classifierType, shadowTrainRaw, shadowTrainLabel, X_test=targetTestRaw, y_test=shadowTestLabel, splitData= False, inepochs=num_epoch, batch_size=100)
	np.savez(attackerModelDataPath + '/shadowModelData.npz', attackModelDataShadow, attackModelLabelsShadow)
	np.savez(modelPath + '/shadowModel.npz', *lasagne.layers.get_all_param_values(shadowModelToStore))
	return attackModelDataShadow, attackModelLabelsShadow

def generateAttackData(dataset, classifierType, dataFolderPath ,pathToLoadData ,num_epoch ,preprocessData ,trainTargetModel ,trainShadowModel,topX=3 ):
	attackerModelDataPath = dataFolderPath+dataset+'/attackerModelData'

	if(preprocessData):
		initializeData(dataset, pathToLoadData, dataFolderPath)

	if(trainTargetModel):
		targetX, targetY = initializeTargetModel(dataset,num_epoch,classifierType =classifierType )
	else:
		targetX, targetY = load_data(attackerModelDataPath + '/targetModelData.npz')

	if(trainShadowModel):
		shadowX, shadowY = initializeShadowModel(dataset,num_epoch,classifierType =classifierType)
	else:
		shadowX, shadowY = load_data(attackerModelDataPath + '/shadowModelData.npz')

	targetX = clipDataTopX(targetX,top=topX)
	shadowX = clipDataTopX(shadowX,top=topX)
	return targetX, targetY, shadowX, shadowY

def attackerOne(dataset= 'CIFAR10',classifierType = 'cnn',dataFolderPath='/content/drive/My Drive/Proposal work/MLeaks/data/',pathToLoadData = '/content/drive/My Drive/Proposal work/MLeaks/data/cifar-10',num_epoch = 50,preprocessData=True,trainTargetModel = True,trainShadowModel=True):
	targetX, targetY, shadowX, shadowY = generateAttackData(dataset,classifierType,dataFolderPath,pathToLoadData,num_epoch,preprocessData,trainTargetModel,trainShadowModel)
	print("Training the attack model for the first adversary")
	trainAttackModel(targetX, targetY, shadowX, shadowY)
def attackerTwo(dataset1= 'CIFAR10',dataset2= 'News',classifierType1 = 'cnn',classifierType2 = 'nn',dataFolderPath='/content/drive/My Drive/Proposal work/MLeaks/data/',pathToLoadData = '/content/drive/My Drive/Proposal work/MLeaks/data/cifar-10',num_epoch = 50,preprocessData=True,trainTargetModel = True,trainShadowModel=True):
	Dataset1X, Dataset1Y, _, _ = generateAttackData(dataset1,classifierType1,dataFolderPath,pathToLoadData,num_epoch,preprocessData,trainTargetModel,trainShadowModel)
	Dataset2X, Dataset2Y, _, _ = generateAttackData(dataset2,classifierType2,dataFolderPath,pathToLoadData,num_epoch,preprocessData,trainTargetModel,trainShadowModel)
	print("Training the attack model for the second adversary")
	trainAttackModel(Dataset1X, Dataset1Y, Dataset2X, Dataset2Y)
    
def attackerThree(dataset= 'CIFAR10',classifierType = 'cnn',dataFolderPath='/content/drive/My Drive/Proposal work/MLeaks/data/',pathToLoadData = '/content/drive/My Drive/Proposal work/MLeaks/data/cifar-10',num_epoch = 50,preprocessData=True,trainTargetModel = True):
	targetX, targetY, _, _ = generateAttackData(dataset,classifierType,dataFolderPath,pathToLoadData,num_epoch,preprocessData,trainTargetModel,trainShadowModel=False,topX=1)
	print('AUC = {}'.format(roc_auc_score(targetY,targetX)))
    
if(opt.adv =='1'):
	#attackerOne(dataset= opt.dataset,classifierType = opt.classifierType,dataFolderPath=opt.dataFolderPath,pathToLoadData = opt.pathToLoadData,num_epoch = opt.num_epoch,preprocessData=opt.preprocessData,trainTargetModel = opt.trainTargetModel, trainShadowModel = opt.trainShadowModel)
	attackerOne(dataset= opt.dataset,classifierType = opt.classifierType,dataFolderPath=opt.dataFolderPath,pathToLoadData = opt.pathToLoadData,num_epoch = opt.num_epoch,preprocessData=True,trainTargetModel = True, trainShadowModel = True)
elif(opt.adv =='2'):
	attackerTwo(dataset1= opt.dataset,dataset2= opt.dataset2,classifierType1 = opt.classifierType,classifierType2 = opt.classifierType2,dataFolderPath=opt.dataFolderPath,pathToLoadData = opt.pathToLoadData,num_epoch = opt.num_epoch, preprocessData = opt.preprocessData, trainTargetModel = opt.trainTargetModel, trainShadowModel = opt.trainShadowModel)

elif(opt.adv =='3'):
	attackerThree(dataset= opt.dataset,classifierType =opt.classifierType,dataFolderPath=opt.dataFolderPath,pathToLoadData = opt.pathToLoadData,num_epoch = opt.num_epoch,preprocessData=opt.preprocessData,trainTargetModel = opt.trainTargetModel)
    
    
    
    
########## Knowledge distilition
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                * self.temperature**2
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
    
# Create the teacher
teacher = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(10),
    ],
    name="teacher",
)

# Create the student
student = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(10),
    ],
    name="student",
)

# Clone student for later comparison
student_scratch = keras.models.clone_model(student)

# Prepare the train and test dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data
x_train = x_train.astype("float32") / 255.0
x_train = np.reshape(x_train, (-1, 28, 28, 1))

x_test = x_test.astype("float32") / 255.0
x_test = np.reshape(x_test, (-1, 28, 28, 1))
