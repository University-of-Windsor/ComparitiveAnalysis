# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:56:53 2023

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

global_optimizer='rmsprop'
global_dataset='fmnist'
global_countermeasure = ['gd']

train_data, train_labels, test_data, test_labels = load_dataset(dataset_name=global_dataset)
cifar_train = train_data, train_labels
cifar_test = test_data, test_labels

cifar_test_data, cifar_test_fed_data, externat_test_data = get_test_data(cifar_test)

teacher_model = create_compiled_keras_model(optimizer=global_optimizer,dataset_name=global_dataset,countermeasures=global_countermeasure) #create_compiled_keras_model()
student_model = create_compiled_keras_model(optimizer=global_optimizer,dataset_name=global_dataset, countermeasures=global_countermeasure) #create_compiled_keras_model()

teacher_model.summary()

student_model.summary()

   
# Train teacher as usual
history_callback = teacher_model.fit(cifar_train_data[0], cifar_train_data[1], validation_data=cifar_test_data, batch_size=32, epochs=12, verbose=1)

# Train and evaluate teacher on data.
teacher_model.evaluate(cifar_test_data[0], cifar_test_data[1])

# Initialize and compile distiller
distiller = Distiller(student=student_model, teacher=teacher_model)
distiller.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    metrics=["accuracy"],
    student_loss_fn=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)

# Distill teacher to student
callback=distiller.fit(cifar_train_data[0], cifar_train_data[1], validation_data=cifar_test_data, batch_size=32, epochs=12, verbose=1)

# Evaluate student on test dataset
distiller.evaluate(cifar_test_data[0], cifar_test_data[1])

#np.mean(callback.history['accuracy'])

# MIA
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
    

###MIA with Prediction sensitivity


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

from utils import *
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
