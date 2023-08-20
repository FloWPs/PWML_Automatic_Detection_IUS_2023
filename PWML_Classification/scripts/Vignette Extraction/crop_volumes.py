#####################
# IMPORT LIBRAIRIES #
#####################


import tensorflow as tf
import numpy as np
import cv2
import glob, os
import hdf5storage
import utils


#####################
#   CONFIGURATION   #
#####################


MODALITY = 'mod-US-9L4'
PATH_CROP_MASK = 'outputs/Crop_masks_3D/' # A SPECIFIER
PATH_CROP_VOLUME = 'outputs/Crop_volumes_3D/' # A SPECIFIER


def patient_format(patient):
    """
    Return patient formated as (Patient-x, J-x_E-x)
    """
    s = patient.split('_')
    p = s[0][:7]+'-'+s[0][7:]
    d = s[1][:1]+'-'+s[1][1:]+'-E-'+s[2]
    return p, d

pat_list = [
    'Patient24_J42_67',
    'Patient25_J33_75',
    'Patient26_J36_70'
]

print("Nombre de patients :", len(pat_list))
# patient = "Patient129_J74_54"

#####################
#    DATA LOADING   #
#####################


for patient in pat_list:
    print('\nPatient à prédire :', patient, '\n')

    p, d = patient_format(patient) # CHEMINS A MODIFIER
    MaskPath = f'PWML_Segmentation/DataManagement/GT expanded/{p}_{MODALITY}_{d}_Repere-commun_PWML_expanded_coronal.mat'
    PatientPath = f'PWML_Segmentation/DataManagement/IMAGES/{p}_{MODALITY}_{d}_Repere-commun.mat'
    # print(PatientPath)
    # print(MaskPath)

    p = patient_format(patient)
    PATH_SAVEVOL = 'PWML_Segmentation/outputs/'+patient+'/volumes/'
    # PATH_SAVEVOL = PATH_CROP_VOLUME+p[0]+'/'+MODALITY+'/'+p[1]+'/'
    if not os.path.isdir(PATH_SAVEVOL):
        os.makedirs(PATH_SAVEVOL)
    
    PATH_SAVEMASK = 'PWML_Segmentation/outputs/'+patient+'/volumes/'
    # PATH_SAVEMASK = PATH_CROP_MASK+p[0]+'/'+MODALITY+'/'+p[1]+'/'
    if not os.path.isdir(PATH_SAVEMASK):
        os.makedirs(PATH_SAVEMASK)


    ######################
    #  CROPPING VOLUMES  #
    ######################

    #----------------
    #  TOP-LEFT 128 
    #----------------

    # Crop parameters to get a 128^3 sub-volume in the top-left part of the volume
    CROP_TL = [150, 189, 139, -22, -61, -11] 
    x_min , y_min, z_min, x_max, y_max, z_max = CROP_TL

    dim_ax = x_max +  x_min
    dim_sag = y_max +  y_min
    dim_cor = z_max +  z_min

    print('\nCrop dimensions :')
    print(x_min, y_min, z_min, x_max, y_max, z_max)
    print('Axial', dim_ax)
    print('Sagittal', dim_sag)
    print('Coronal', dim_cor)

    # Load Patient
    # Raw mat file -> cropped volume
    reduced_Volume, Initial_volume = utils.preprocessing_prediction(PatientPath, CROP_TL)
    # print("reduced volume :", np.max(reduced_Volume))

    # Cast to float32
    reduced_Volume = tf.cast(reduced_Volume, tf.float32)
    np.save(os.path.join(PATH_SAVEVOL, 'crop_input_TL_128.npy'), reduced_Volume.numpy()) # A SPECIFIER
    
    # Load groundtruth mask
    reduced_mask, Initial_mask = utils.preprocessing_prediction(MaskPath, CROP_TL, mask=True, lesion_percentage=0.9)
    np.save(os.path.join(PATH_SAVEMASK, 'crop_mask_TL_exp_L90_128.npy'), reduced_mask.numpy()) # A SPECIFIER
    

    #-----------------
    #  TOP-RIGHT 128 
    #-----------------

    # Crop parameters to get a 128^3 sub-volume in the top-right part of the volume
    CROP_TR = [139, -61, 149, -11, 189, -21]
    x_min , y_min, z_min, x_max, y_max, z_max = CROP_TR

    dim_ax = x_max +  x_min
    dim_sag = y_max +  y_min
    dim_cor = z_max +  z_min

    print('\nCrop dimensions :')
    print(x_min, y_min, z_min, x_max, y_max, z_max)
    print('Axial', dim_ax)
    print('Sagittal', dim_sag)
    print('Coronal', dim_cor)

    # Load Patient
    # Raw mat file -> cropped volume
    reduced_Volume, Initial_volume = utils.preprocessing_prediction(PatientPath, CROP_TR)
    # print("reduced volume :", np.max(reduced_Volume))

    # Cast to float32
    reduced_Volume = tf.cast(reduced_Volume, tf.float32)
    np.save(os.path.join(PATH_SAVEVOL, 'crop_input_TR_128.npy'), reduced_Volume.numpy()) # A SPECIFIER
    
    # Load groundtruth mask
    reduced_mask, Initial_mask = utils.preprocessing_prediction(MaskPath, CROP_TR, mask=True, lesion_percentage=0.9)
    np.save(os.path.join(PATH_SAVEMASK, 'crop_mask_TR_exp_L90_128.npy'), reduced_mask.numpy()) # A SPECIFIER