#######################
#  IMPORT LIBRAIRIES  #
#######################


import os
import glob
import shutil

import random
import datetime
import time as time
import numpy as np
import pandas as pd

import imageio
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from keras.preprocessing.image import img_to_array

from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import tensorflow_addons as tfa
from model import new_model
from tensorflow.keras.optimizers import SGD, Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau

from metrics import recall_m, precision_m, f1_m, MCC, get_cv_scores
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')



#######################
#     DATA LOADING    #
#######################


def patient_format(patient, modality='mod-US-9L4'):
    """
    Return correct patient name to retrieve data afterwards.
    """
    p = patient.split('_')
    patientID = f'{p[0][:7]}-{p[0][7:]}'
    exam = f'{p[1][:1]}-{p[1][1:]}-E-{p[2]}'

    return f'{patientID}_{modality}_{exam}'


def dataLoader(datapath, patlist, label):
    """
    Return all images of patlist from the dataset and generate the corresponding labels.
    """
    images = []
    labels = []

    for p in patlist :
        # MODIFIER DIMENSION ENTRAINEMENT
        for pid in glob.iglob(datapath+patient_format(p)+'/*.npy', recursive=True):
            # Image loading
            image = np.load(pid)[:, :, 1:]
            image = image.astype("float32")
            # Normalization
            image /= 255
            # apply the Keras utility function that correctly
            # rearranges the dimensions of the image
            image = img_to_array(image)

            images.append(image)
            labels.append(label)
    
    return images, labels


#######################
#    CONFIGURATION    #
#######################


MODALITY = 'mod-US-9L4'

# Datasets paths # MOFIDIER LES JEUX DE DONNEES
PATH_FALSE_ALARM_PU = 'D:/PWML_Classification/Datasets/False_alarms_3D_PU_expanded_L100_32_v2/'
PATH_TRUE_ALARM_TR = 'D:/PWML_Classification/Datasets/True_alarms_3D_expanded_L90_TR_ALL_32/'
PATH_TRUE_ALARM_TL = 'D:/PWML_Classification/Datasets/True_alarms_3D_expanded_L90_TL_ALL_32/'

DATA_SOURCES =  [[PATH_FALSE_ALARM_PU, 0], [PATH_TRUE_ALARM_TR, 1], [PATH_TRUE_ALARM_TL, 1]]

PATLIST = [
    'Patient24_J42_67',
    'Patient25_J33_75',
    'Patient26_J36_70'
]

# Model Hyperparameters
WINDOW_SIZE = 32
CHANNELS = 2
LR = 0.001
BATCHSIZE = 32
EPOCHS = 200

OUTPUT = '../outputs/models/multi-crossval'
if not os.path.isdir(OUTPUT):
    os.makedirs(OUTPUT)

# Number of cross-validation runs
NB_RUNS = 10



#################################
#    K-FOLD CROSS-VALIDATION    #
#################################


t = time.time()
dt = datetime.datetime.now()
direc = f'2.5D-SC-Net_{dt.year}-{dt.month}-{dt.day}_{dt.hour}-{dt.minute}'

run_infos = []

# Multi rounds of k-fold cross-validation
for r in range(NB_RUNS):
    tr = time.time()
    rinfos = dict({'Run': r})
    print(f'\n[INFO] STARTING RUN n°{r}')

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=10, shuffle=True)
    kfold_infos = []

    # K-fold Cross Validation
    for k, data in enumerate(kfold.split(PATLIST)):
        tk = time.time()

        # Record informations about the fold
        TRAIN_PATLIST = np.array(PATLIST)[data[0]]
        VALID_PATLIST = np.array(PATLIST)[data[1]]
        infos = dict({'K-fold': k, 'Train Patients': TRAIN_PATLIST, 'Valid Patients': VALID_PATLIST})
        print(f'\n[INFO] KFOLD n°{k}')
        # print(f'Number of training patients : {len(TRAIN_PATLIST)}')
        # print(f'Number of valid patients : {len(VALID_PATLIST)}')

        # Data Loading
        x_train, y_train = [], []
        x_valid, y_valid = [], []

        for source in DATA_SOURCES:
            # Train dataset
            x, y = dataLoader(source[0], TRAIN_PATLIST, label=source[1])
            x_train += x
            y_train += y

            # Valid dataset
            x, y = dataLoader(source[0], VALID_PATLIST, label=source[1])
            x_valid += x
            y_valid += y
        
        # Array conversion
        x_train = np.array(x_train)
        x_valid = np.array(x_valid)

        infos.update({'Number of train images': x_train.shape[0]})
        infos.update({'Number of valid images': x_valid.shape[0]})
        infos.update({'Vignette dimensions': x_train.shape[1:]})
        print(f"[INFO] Vignettes dimensions : {x_train.shape[1:]}")

        # Calculate the weights for each class so that we can balance the data
        weights = class_weight.compute_class_weight('balanced',
                                                    classes = np.unique(y_train),
                                                    y = y_train)
        # weights = dict(zip(np.unique(y_train), weights))
        weights = {i:w for i,w in enumerate(weights)}
        infos.update({'Class weights': weights})

        # Encode the labels to [0 1] or [1 0]
        y_train = np_utils.to_categorical(LabelBinarizer().fit_transform(y_train), 2)  # vecteur de taille 2 pour chaque label
        y_valid = np_utils.to_categorical(LabelBinarizer().fit_transform(y_valid), 2)
        infos.update({'Train images with PWML': len(y_train[y_train[:, 1]==1])})
        infos.update({'Valid images with PWML': len(y_valid[y_valid[:, 1]==1])})

        x_train, y_train = shuffle(x_train, y_train)
        x_valid, y_valid = shuffle(x_valid, y_valid)
        print(f"[INFO] DATA : training with {len(x_train)} images, validating on {len(x_valid)} images")
        
        # Model Building
        print("[INFO] Intitializing and compiling the model...")
        
        model = new_model(WINDOW_SIZE, CHANNELS)
        
        # opt = SGD(lr=LR, momentum=.9)
        opt = Adam(lr=LR)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["acc",
                                                                        tf.keras.metrics.Recall(class_id=1),
                                                                        tf.keras.metrics.Precision(class_id=1),
                                                                        tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)
                                                                    ])
        
        # Create the output directory
        output_file = os.path.join(OUTPUT, direc, 'run-'+str(r), 'kfold-'+str(k))
        if os.path.exists(output_file):
            shutil.rmtree(output_file)
        os.makedirs(output_file)

        # callback configuration
        print("[INFO] Preparing the callbacks...")
        callbacks = [
            ModelCheckpoint(output_file + f"/PWML_classification_model_kfold_{k}.tf", monitor="val_loss", mode="min",
                            save_best_only=True, verbose=1),
            # TrainingMonitor(output_file + "/plot.png"),
            CSVLogger(output_file + "/training_log_kfold_"+str(k)+".csv"),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1),
            EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
        
        # construct the image generator for data augmentation
        print("[INFO] Using real-time data augmentation...")
        aug = ImageDataGenerator(rotation_range=30, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1,
                                shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
        aug.fit(x_train)
        train_generator = aug.flow(x_train, y_train, batch_size=BATCHSIZE, seed=1)

        # Training the Model
        print("[INFO] Training the model...")
        model.fit_generator(train_generator, steps_per_epoch=len(x_train) // BATCHSIZE,
                            validation_data=(x_valid, y_valid), shuffle=True,
                            epochs=EPOCHS, callbacks=callbacks, verbose=1, class_weight=weights)

        infos.update({'Model path': output_file + f"/PWML_classification_model_kfold_{k}.tf"})

        # Print elapsed time
        elapsed = round((time.time() - tk) / 60, 2)
        infos.update({'Elapsed time (min)': elapsed})
        print(f"[INFO] Elapsed time is {elapsed} minutes")

        # Save predictions for later evaluation
        logits = model.predict(x_valid)
        y_pred = tf.argmax(input=logits, axis=1)
        y_true = tf.argmax(input=y_valid, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        infos.update({'Confusion Matrix': cm})

        # Add new row to dataframe
        kfold_infos.append(infos)

    df = pd.DataFrame(kfold_infos, columns=['K-fold', 'Train Patients', 'Valid Patients', 'Number of train images', 'Train images with PWML', 'Number of valid images', 'Valid images with PWML', 'Class weights', 'Vignette dimensions', 'Model path', 'Confusion Matrix', 'Elapsed time (min)'])
    df.to_csv(os.path.join(OUTPUT, direc, 'run-'+str(r), 'run-'+str(r)+'_kfolds_config.csv'), index=False)

    # Compute metrics on all the k-folds
    tcm = np.zeros((2,2))
    for j in range(len(df)):
        tcm = tcm + df.loc[j, 'Confusion Matrix']
    
    rinfos.update({'Confusion Matrix': tcm})
    scores = get_cv_scores(tcm)
    rinfos.update(scores)

    elapsed = round((time.time() - tr) / 60, 2)
    rinfos.update({'Elapsed time (min)': elapsed})
    print(f"\n[INFO] Total elapsed time for run {r} is {elapsed} minutes")

    # Add new row to dataframe
    run_infos.append(rinfos)

elapsed = round((time.time() - t) / 60, 2)
print(f"\n[INFO] Total elapsed time for the 10 cross-validations is {elapsed} minutes")

dft = pd.DataFrame(run_infos, columns=['Run', 'Confusion Matrix', 'Accuracy', 'Recall', 'Precision', 'MCC', 'Elapsed time (min)'])

dft.to_csv(os.path.join(OUTPUT, direc, 'all_kfolds_results_3D-SC_exp_dnsd_L100_with_FP_PU_ASC_vignettes_32.csv'), index=False) # A MODIFIER FILENAME
