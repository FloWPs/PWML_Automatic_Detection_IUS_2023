#######################
#  IMPORT LIBRAIRIES  #
#######################

import os
import numpy as np
import pandas as pd
import cc3d

from metrics import get_scores

import tensorflow as tf
import tensorflow_addons as tfa
from keras.models import load_model
from keras.preprocessing.image import img_to_array

from processing import extract_vignette, remove_small_lesions

from tqdm import tqdm


#####################
#   CONFIGURATION   #
#####################


# Datapaths
PATH_VOLUMES = 'D:/IUS_2023/PWML_Segmentation/outputs/'

# Models path
PATH_MODEL_3D = 'D:/IUS_2023/PWML_Classification/outputs/models/multi-crossval/2.5D-SC-Net_2023-3-3_14-17/run-0/kfold-1/PWML_classification_model_kfold_1.tf'


# List of test patients
PATLIST = [
    'Patient128_J41_72',
    'Patient129_J74_54',
    'Patient130_J29_47'
]

# Threshold
t = 0.1

# Vignette dimensions
VIGNETTE_DIM = (32, 32) # (16, 16) # (64, 64)

# Number of channels
CHANNELS = 2 # MODIFY NUMBER OF CHANNELS

row_list = []

for patient in tqdm(PATLIST):

    PATH_PAT = os.path.join(PATH_VOLUMES, patient, 'TU_correction_pred')
    if not os.path.isdir(PATH_PAT):
            os.makedirs(PATH_PAT)
    

    #####################
    #    DATA LOADING   #
    #####################

    # Load volume
    volume = np.load(os.path.join(PATH_VOLUMES, patient, 'volumes', 'crop_input_L90_128.npy'))
    volume = volume.astype("float32")
    # Normalization
    volume /= 255

    # Load grountruth
    gt_mask = np.load(os.path.join(PATH_VOLUMES, patient, 'volumes', 'crop_mask_corrige_L90_128.npy'))
  
    # Load TransUNet Prediction
    PU_mask = np.load(os.path.join(PATH_VOLUMES, patient, 'volumes', 'coronal_pred_0.1.npy')) # Coronal prediction from TransUNet
    PU_mask = PU_mask*1


    ######################
    #   DATA PROCESSING  #
    ######################
    
    #--------------------
    #   Models Loading
    #--------------------

    model_3D = load_model(PATH_MODEL_3D)

    #----------------------------------------------------
    #    Connected Component Extraction & Prediction
    #----------------------------------------------------

    PU_mask_correction = PU_mask.copy()

    # apply connected component analysis to the thresholded image
    connectivity = 26 # 3D
    output, N = cc3d.connected_components(PU_mask, connectivity=connectivity, return_N=True)
    # If there is no lesion
    # if N == 0:
        # print('No lesions found.')

    # Get number of voxels, bounding box and centroid for each lesion
    stats = cc3d.statistics(output)

    # Number of voxels per lesion cluster with segid as key
    lesions = dict(enumerate(stats['voxel_counts']))
    # print(lesions)
    # First element is always the background
    del lesions[0]

    # List of centroids per lesion cluster with segid as key
    centroids = dict(enumerate(stats['centroids']))
    # print(lesions)
    # First element is always the background
    del centroids[0]

    for k in range(1, len(lesions)+1):
        # print(lesions[k], centroids[k])
        if lesions[k] > 0:
            # vignetteA = extract_vignette(volume[int(centroids[k][0]), :, :], [int(centroids[k][1]), int(centroids[k][2])], dimensions=VIGNETTE_DIM)
            vignetteS = extract_vignette(volume[:, int(centroids[k][1]), :], [int(centroids[k][0]), int(centroids[k][2])], dimensions=VIGNETTE_DIM)
            vignetteC = extract_vignette(volume[:, :, int(centroids[k][2])], [int(centroids[k][0]), int(centroids[k][1])], dimensions=VIGNETTE_DIM)
            
            # COMMENT OR UNCOMMENT DEPENDING ON THE NUMBER OF CHANNELS
            # # Concatenate the results into a single image with 3 channels ASC
            # vignette_3D = np.zeros((VIGNETTE_DIM[0], VIGNETTE_DIM[1], CHANNELS), dtype=np.float32)
            # vignette_3D[:, :, 0] = vignetteA
            # vignette_3D[:, :, 1] = vignetteS
            # vignette_3D[:, :, 2] = vignetteC

            # Concatenate the results into a single image with 3 channels SC
            vignette_3D = np.zeros((VIGNETTE_DIM[0], VIGNETTE_DIM[1], CHANNELS), dtype=np.float32)
            vignette_3D[:, :, 0] = vignetteS
            vignette_3D[:, :, 1] = vignetteC

            # Make label prediction with the classification network
            vignette_3D = vignette_3D.reshape(1, VIGNETTE_DIM[0], VIGNETTE_DIM[1], CHANNELS)
            y_pred = tf.argmax(model_3D.predict(vignette_3D), axis=1)

            # Modify the mask predicted with Priority U-Net
            if y_pred[0] == 0:
                lesion_cluster = np.where(output == k)
                PU_mask_correction[lesion_cluster] = 0
            
    # Save new prediction (PU mask with correction) # MODIFIER NOM
    np.save(os.path.join(PATH_PAT, 'pred_correction_L100_exp_dnsd_PU-C_3D_dnsd-SC_FP-ASC_32_'+str(t)+'.npy'), PU_mask_correction) # MODIFIER DIMENSION

    #----------------------
    #    New Evaluation
    #----------------------

    new_scores = get_scores(patient, t, 'TransUnet after classif 2.5D S+C', gt_mask, PU_mask_correction) # PU_mask_correction) # MODIFIER DIMENSION
    row_list.append(new_scores)

#####################
#  SAVE THE RESULTS #
#####################

df = pd.DataFrame(row_list, columns=['PatientID', 'Threshold', 'Training Dim', 'Dice',
    'Recall (Lesion-wise)', 'Precision (Lesion-wise)', 'F1-Score (Lesion-wise)', 'True Intersection (Pixel-wise)', 'FN (Pixel-wise)', 'TP (Pixel-wise)',
    'MCC (Slice-wise)', 'Recall (Slice-wise)', 'Precision (Slice-wise)', 'F1-Score (Slice-wise)', 'Accuracy (Slice-wise)', 'TN', 'FP', 'FN', 'TP', 'FN_pred', 'TP_pred'])

# df.to_csv('L100_TOTAL_RESULTS_C_AFTER_CLASSIF_3D_SC_32.csv', index=False) # MODIFIER NOM