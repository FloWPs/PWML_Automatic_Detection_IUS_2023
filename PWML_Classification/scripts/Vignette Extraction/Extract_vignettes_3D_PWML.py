#####################
# IMPORT LIBRAIRIES #
#####################


import os, glob
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2
import cc3d

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


PATH_CROP_VOLUME = 'PWML_Segmentation/outputs/'  
PATH_CROP_MASK = 'PWML_Segmentation/outputs/'

VIGNETTE_DIM =(32, 32) # (16, 16) #(64 ,64) #

# Extract PWML from 2 sub regions of the brain
for POS in ['TL', 'TR']: 

    PATH_TRUE_ALARM = f'PWML_Classification/Datasets/True_alarms_3D_exp_L90_{POS}_32_v0/' # MODIFIER PATH
    if not os.path.isdir(PATH_TRUE_ALARM):
        os.makedirs(PATH_TRUE_ALARM)

    # Number of vignette generated
    row_list = []


    #####################
    #    DATA LOADING   #
    #####################

    VIGNETTE_DIM = (32, 32) # (16, 16) #

    # Number of vignette generated
    tot_vignettes = 0
    row_list = []

    for mname in glob.iglob(PATH_CROP_VOLUME + f'**/volumes/*crop_mask_{POS}_exp_L90_128.npy', recursive=True):

        patient = mname.split('\\')[1]
    #     exam = df.loc[i, 'Exam']
        print(patient)

        # load groundtruth
        mask = np.load(mname)
        print(mask.shape, np.max(mask))

        # load volume
        vname = mname.replace(f'mask_{POS}_exp_L90', f'input_{POS}')
        volume = np.load(vname)

        PATH_PAT = PATH_TRUE_ALARM + patient +'/'
        if not os.path.isdir(PATH_PAT):
            os.makedirs(PATH_PAT)

        PATH_VIGNETTES = PATH_PAT #+ '3D/'
        if not os.path.isdir(PATH_VIGNETTES):
            os.makedirs(PATH_VIGNETTES)


        ########################
        #    DATA PROCESSING   #
        ########################

        #----------------------
        #  Dimension coronale
        #----------------------

        cc = 0
        for c in range(volume.shape[2]): #[70]:#
            # element_image = volume[:, :, c]
            element_mask = mask[:, :, c]

            if np.max(element_mask) > 0:
                # Get centroids of PWML on the coronal slice (A + S coordinates)
                centroids = get_centroids(element_mask, cc_area_min=0) # MODIFIER CC_AREA_MIN (0 si masque corrigé, 20 px sinon)

                for j in range(len(centroids)):
                    a = int(centroids[j][1]) # Order is reversed
                    s = int(centroids[j][0])
                    # print(a, s, c)
                    # Get the 3 corresponding A+S+C orthogonal slices
                    element_image_A = volume[a, :, :]
                    element_image_S = volume[:, s, :]
                    element_image_C = volume[:, :, c]

                    # Extract the 3 corresponding A+S+C orthogonal vignettes
                    vignette_A = extract_vignette(element_image_A, [s, c], dimensions=VIGNETTE_DIM)
                    vignette_S = extract_vignette(element_image_S, [a, c], dimensions=VIGNETTE_DIM)
                    vignette_C = extract_vignette(element_image_C, [a, s], dimensions=VIGNETTE_DIM)

                    # Concatenate the results into a single image with 3 channels
                    vignette_3D = np.zeros((VIGNETTE_DIM[0], VIGNETTE_DIM[1], 3), dtype=np.float32)
                    vignette_3D[:, :, 0] = vignette_A
                    vignette_3D[:, :, 1] = vignette_S
                    vignette_3D[:, :, 2] = vignette_C

                    # # Ensure that there are not too many adjacent slices in the database
                    # if random.randint(0, 1):
                    # Extract 3 orthogonal slices centered around the connected component
                    # Save image
                    np.save(os.path.join(PATH_VIGNETTES+'/TA_3D_'+str(c)+'_cc-'+str(j+1)+'.npy'), vignette_3D.astype(np.float32))
                    cc += 1
        

        # Monitoring of the number of vignettes generatad for each patient
        stats = {'PatientID': patient, 'Nombre de vignettes 3D': cc}
        # print(stats)
        row_list.append(stats)
        tot_vignettes = tot_vignettes + cc

    print(f'\nTotal number of 3D vignettes generated for {POS} 128 : {tot_vignettes}\n')

    df_stats = pd.DataFrame(row_list, columns=['PatientID', 'Nombre de vignettes 3D'])
    # df_stats.to_csv('vignettes_3D_ALL_PWML_extraction_stats_exp_dnsd_L90_TL_32.csv', index=False) # MODIFIER NOM CSV