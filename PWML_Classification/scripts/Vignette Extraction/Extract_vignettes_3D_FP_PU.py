#####################
# IMPORT LIBRAIRIES #
#####################


import os
import glob

import random
# import imageio
# import hdf5storage 
import pandas as pd
import numpy as np

import cv2
# import cc3d

# from tqdm import trange



#######################################
#    CONNECTED COMPONENT EXTRACTION   #
#######################################


def get_centroids(mask, cc_area_min=20):
    """
    Return centroids coordinates from connected components bigger than cc_area_min (in pixels)
    """
    c = []
    # apply connected component analysis to the thresholded image
    connectivity = 8
    output = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    
    # loop over the number of unique connected component labels, skipping
    # over the first label (as label zero is the background)
    for i in range(1, numLabels):
        # extract the connected component statistics for the current# label
        area = stats[i, cv2.CC_STAT_AREA]
        # print('Connected component', i, ':', area, 'pixels')
        # ensure area is not too small
        keepArea = area > cc_area_min
        if keepArea:
            # add centroid to the list
            c.append(centroids[i])
            # print("[INFO] keeping connected component '{}'".format(i))
    
    return c


def extract_vignette(image, centroid, dimensions=(64, 64)):
    """
    Extract a vignette of given dimensions from an image
    centered around the centroid coordinates (connected component)
    """
    # print(centroid[1], centroid[0])
    # x1 = int(centroid[1])-(dimensions[0]//2)
    # x2 = int(centroid[1])+(dimensions[0]//2)
    # y1 = int(centroid[0])-(dimensions[1]//2)
    # y2 = int(centroid[0])+(dimensions[1]//2)
    x1 = int(centroid[0])-(dimensions[0]//2)
    x2 = int(centroid[0])+(dimensions[0]//2)
    y1 = int(centroid[1])-(dimensions[1]//2)
    y2 = int(centroid[1])+(dimensions[1]//2)
    # print(x1, x2, y1, y2)

    # Dimension verification
    if (x2-x1) != dimensions[0]:
        d = dimensions[0] - (x2-x1)
        x2 += d
    if (y2-y1) != dimensions[1]:
        d = dimensions[1] - (y2-y1)
        y2 += d

    # If vignette out of bounds
    if x1 < 0:
        # print(1)
        x2 -= x1
        x1 -= x1
    if x2 > image.shape[0]:
        # print(2)
        x1 += (image.shape[0])-x2
        x2 += (image.shape[0])-x2
    if y1 < 0:
        # print(3)
        y2 -= y1
        y1 -= y1
        
    if y2 > image.shape[1]:
        # print(4)
        y1 += (image.shape[1])-y2
        y2 += (image.shape[1])-y2
    
    vignette = image[x1:x2, y1:y2]
    # plt.imshow(vignette, 'gray')
 
    assert vignette.shape == dimensions

    return vignette



#####################
#   CONFIGURATION   #
#####################


MODALITY = 'mod-US-9L4'

# Datapaths
PATH_VOLUMES = 'PWML_Segmentation/outputs/'

PATH_FALSE_ALARM = 'PWML_Classification/Datasets/False_alarms_3D_exp_L100_32_v0/' # MODIFIER PATH
if not os.path.isdir(PATH_FALSE_ALARM):
    os.makedirs(PATH_FALSE_ALARM)


PATLIST = [
    'Patient24_J42_67',
    'Patient25_J33_75',
    'Patient26_J36_70'
]

def get_infos(patient):
    """
    Return patient at format ('Patient-x4', 'J-x-E-x')
    """
    p = patient.split('_')
    pat = ('-').join([p[0][:7], p[0][7:]])
    exam = ('-').join([p[1][:1], p[1][1:], 'E', p[2]])

    return pat, exam

# Threshold
t = 0.1

# Vignette dimensions
VIGNETTE_DIM = (32, 32) #(16, 16) #(64, 64)

# Number of vignette generated
row_list = []



#####################
#    DATA LOADING   #
#####################

tot_vignettes = 0

for i in range(len(PATLIST)): #[10]: #

    patient = PATLIST[i]

    # load volume
    volume = np.load(os.path.join(PATH_VOLUMES, patient, 'volumes', 'crop_input_TR_128.npy')) # MODIFIER INPUT

    # load groundtruth
    gt_mask = np.load(os.path.join(PATH_VOLUMES, patient, 'volumes', 'crop_mask_TR_exp_L90_128.npy')) # MODIFIER INPUT

    # threshold prediction
    t = 0.1

    # load predictions
    PU_mask = np.load(os.path.join(PATH_VOLUMES, patient, 'PU_multimodal_pred', 'pred_multi_ASC_expanded_'+str(t)+'.npy')) # MODIFIER INPUT

    #--------------------
    #  Paths for saving
    #--------------------

    pat, exam = get_infos(patient)

    PATH_PAT = PATH_FALSE_ALARM + patient +'/'
    if not os.path.isdir(PATH_PAT):
        os.makedirs(PATH_PAT)


    ########################
    #    DATA PROCESSING   #
    ########################

    #----------------------
    #  Dimension coronale
    #----------------------

    cc = 0

    # Dimension coronale
    for c in range(PU_mask.shape[2]): #68, 70):#
        image = volume[:, :, c]
        gt = gt_mask[:, :, c]
        pred = PU_mask[:, :, c]

        if np.max(PU_mask) > 0:
            # Get centroids of false alarms
            centroids = get_centroids(pred, cc_area_min=0)

            for j in range(len(centroids)):
                # Get the 3 corresponding A+S+C orthogonal slices
                a = int(centroids[j][1]) # Order is reversed
                s = int(centroids[j][0])
                # print(a, s, c)
                element_image_A = volume[a, :, :]
                element_image_S = volume[:, s, :]
                element_image_C = volume[:, :, c]
                # # Ensure that there are not too many adjacent slices in the database
                if random.randint(0, 1):
                # if 1+1 == 2:
                # if random.randint(0, 2) == 2: # A MODIFIER
                    # Extract a reduced image centered around the connected component
                    # Extract the 3 corresponding A+S+C orthogonal vignettes
                    vignette_A = extract_vignette(element_image_A, [s, c], dimensions=VIGNETTE_DIM)
                    vignette_S = extract_vignette(element_image_S, [a, c], dimensions=VIGNETTE_DIM)
                    vignette_C = extract_vignette(element_image_C, [a, s], dimensions=VIGNETTE_DIM)

                    mask_A = extract_vignette(gt_mask[a, :, :], [s, c], dimensions=VIGNETTE_DIM)
                    mask_S = extract_vignette(gt_mask[:, s, :], [a, c], dimensions=VIGNETTE_DIM)
                    mask_C = extract_vignette(gt_mask[:, :, c], [a, s], dimensions=VIGNETTE_DIM)


                    if np.max(mask_A)<1 and np.max(mask_S)<1 and np.max(mask_C)<1:
                        # Concatenate the results into a single image with 3 channels
                        vignette_3D = np.zeros((VIGNETTE_DIM[0], VIGNETTE_DIM[1], 3), dtype=np.float32)
                        vignette_3D[:, :, 0] = vignette_A
                        vignette_3D[:, :, 1] = vignette_S
                        vignette_3D[:, :, 2] = vignette_C
                        # print('KEEP')

                        # Save 3D vignette
                        np.save(os.path.join(PATH_PAT+'/FA_3D_PU_'+str(c)+'_cc-'+str(j+1)+'.npy'), vignette_3D.astype(np.float32))
                        cc += 1


    # Monitoring of the number of vignettes generatad for each patient
    stats = {'PatientID': patient, 'Nombre de vignettes 3D': cc}
    print(stats)
    row_list.append(stats)

    tot_vignettes = tot_vignettes + cc

print(f'Total number of vignettes generated : {tot_vignettes}')

df_stats = pd.DataFrame(row_list, columns=['PatientID', 'Nombre de vignettes 3D'])
# df_stats.to_csv('vignettes_FP_PU_exp_dnsd_L100_extraction_stats_32.csv', index=False)

