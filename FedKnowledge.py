# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 13:22:56 2023

@author: aliab
"""
import numpy as np
import tensorflow as tf
import random
import datetime
#import tensorflow_privacy
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from utils import *
from Distiller import  *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--optimizer',  default='sgd', help='which is optimizer that can be "sgd,adadelta,adagrad,nadam,rmsprop"')
parser.add_argument('--dataset',  default='mnist', help='which is the dataset that can be one of "mnist, fmnist, cifar10"')
parser.add_argument('--measure',  default='gn', help='which is the countermeasure that can be one of the values "d, mcd, ar, gn, gd, bn, m"')

opt = parser.parse_args()

global_optimizer = opt.optimizer
global_dataset= opt.dataset
global_countermeasure = [opt.measure]

train_data, train_labels, test_data, test_labels = load_dataset(dataset_name=global_dataset)

cifar_train = train_data, train_labels
cifar_test = test_data, test_labels

cifar_train_data, cifar_train_fed_data, attacker_data = get_data(cifar_train)



cifar_test_data, cifar_test_fed_data, externat_test_data = get_test_data(cifar_test)

teacher_model = create_compiled_keras_model(optimizer=global_optimizer,dataset_name=global_dataset,countermeasures=['d']) #create_compiled_keras_model()
student_model = create_compiled_keras_model(optimizer=global_optimizer,dataset_name=global_dataset,countermeasures=global_countermeasure)#create_compiled_keras_model()

teacher_model.summary()
student_model.summary()
   
initial_model = create_compiled_keras_model(optimizer=global_optimizer,dataset_name=global_dataset)#create_compiled_keras_model()

FedTrain = []
FedTest = []
FedAcc = []

DistTrain = []
DistTest = []
DistAcc = []

Dstudent_loss = []
Ddistillation_loss = []
Daccuracy = []

CLIENTS=3
for r in range(6):
    
    deltas = []

    for c in range(CLIENTS):

        teacher_model = create_compiled_keras_model(optimizer=global_optimizer,dataset_name=global_dataset,countermeasures=global_countermeasure)#create_compiled_keras_model()

        teacher_model.set_weights(initial_model.get_weights())

        #fed_history_callback = federated_model.fit(cifar_train_fed_data[c][0], cifar_train_fed_data[c][1], 
         #                                      batch_size=250, epochs=10, verbose=1)
        fed_history_callback = teacher_model.fit(cifar_train_fed_data[c][0], cifar_train_fed_data[c][1], 
                                               batch_size=32, epochs=10, verbose=1)
        
        delta = np.array(initial_model.get_weights()) - np.array(teacher_model.get_weights())

        deltas.append(delta)

    print('Epoch {}/18'.format(r+1))
    delt_av = (deltas[0] + deltas[1] + deltas[2]) / 3
    new_weights = np.array(initial_model.get_weights()) - delt_av
    initial_model.set_weights(new_weights)
    
    FedTrain.append(initial_model.evaluate(cifar_train_data[0], cifar_train_data[1])[0])
    validation = initial_model.evaluate(cifar_test_data[0], cifar_test_data[1])
    FedTest.append(validation[0])
    FedAcc.append(validation[1])

    distiller = Distiller(student=student_model, teacher=teacher_model)
    distiller.compile(
    optimizer=keras.optimizers.SGD(),
    metrics=["accuracy"],
    student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
    )
    # Distill teacher to student
    dist = distiller.fit(cifar_train_data[0], cifar_train_data[1], epochs=10)
    Dstudent_loss.append(dist.history['student_loss'])
    Ddistillation_loss.append(dist.history['distillation_loss'])
    Daccuracy.append(dist.history['accuracy'])

    # Evaluate student on test dataset
    #distiller.evaluate(cifar_test_data[0], cifar_test_data[1])

    DistTrain.append(distiller.evaluate(cifar_train_data[0], cifar_train_data[1])[0])
    validation = distiller.evaluate(cifar_test_data[0], cifar_test_data[1])
    DistTest.append(validation[0])
    DistAcc.append(validation[1])
    
np.mean(dist.history['student_loss'])
np.mean(dist.history['distillation_loss'])
np.mean(dist.history['accuracy'])
np.mean(DistTrain)
np.mean(DistTest)
np.mean(DistAcc)
np.mean(Dstudent_loss)
np.mean(Ddistillation_loss)
global_accuracy=np.mean(Daccuracy)

print("global accuracy: ", global_accuracy )









import numpy as np

from absl import app
from absl import flags

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data
from MiaUtils import *

NUM_CLASSES = 10

SHADOW_DATASET_SIZE = 1000
ATTACK_TEST_DATASET_SIZE = 5000

num_shadows = 10
#num_shadows = 1

target_model = student_model
def target_model_fn():
    return create_compiled_keras_model(optimizer=global_optimizer,dataset_name=global_dataset,countermeasures=global_countermeasure)
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
        epochs=32,
        verbose=True,
        validation_data=(attacker_X_test, attacker_y_test)
    )
)
# ShadowModelBundle returns data in the format suitable for the AttackModelBundle.
amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)

# Fit the attack models.
print("Training the attack models...")
amb.fit(X_shadow, y_shadow, fit_kwargs=dict(epochs=32, verbose=True)
)

target_data = cifar_train_fed_data[0]
attacker_data = cifar_test_fed_data[0]

#attacker_data = cifar_train_data
attack_test_data, real_membership_labels = prepare_attack_data(target_model, target_data, attacker_data)
attack_guesses = amb.predict(attack_test_data)
attack_precision = np.mean((attack_guesses == 1) == (real_membership_labels == 1))

class_precision = []

for c in range(NUM_CLASSES):
    #attack_test_data, real_membership_labels = prepare_attack_data(centralized_model, cifar_train_data, attacker_data)
    target_indices = [i for i, d in enumerate(target_data[1].argmax(axis=1)) if d == c]
    test_indices = [i for i, d in enumerate(attacker_data[1].argmax(axis=1)) if d == c]


    print(np.sum(attack_guesses[target_indices]==1) / (np.sum(attack_guesses[target_indices]) + np.sum(attack_guesses[1000:][test_indices])))
    
    class_precision.append(
            np.sum(attack_guesses[target_indices]==1) / (np.sum(attack_guesses[target_indices])
                                                     + np.sum(attack_guesses[1000:][test_indices])))
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

target_model = student_model
#target_model = single_model0

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
    "mnist": [0,1,2,3,4,5,6,7,8,9],
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

print("average")
print(sum(precisions)/len(precisions), sum(recalls)/len(recalls), sum(f1_scores)/len(f1_scores))