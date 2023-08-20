#---------------------------------------------------------------------------
# The goal of this file is to preprocess the date in order to get 2D
# slices containing PWML for the training of the model.
# The volumes are first crop and then sliced along the coronal projection.
#---------------------------------------------------------------------------

#####################
# IMPORT LIBRAIRIES #
#####################

import scipy.io
import numpy as np
import nibabel as nib
import glob, os
import hdf5storage 
import re
import time, sys
from IPython.display import clear_output
import os
#from cc3d import connected_components
import cv2
import cc3d
import shutil



############################
# REMOVAL OF SMALL LESIONS #
############################

def vox2mm3(vol):
    """
    Convert number of voxels to volume in mm^3
    """
    voxel_volume = 0.15**3
    return vol * voxel_volume

def remove_small_lesions(mask, percentage=0.9):
    """
    Only keep 90% of the biggest lesions from a segmentation map (3D Mask)
    """
    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    lc = 0
    # apply connected component analysis to the thresholded image
    connectivity = 26 # 3D
    output, N = cc3d.connected_components(mask, connectivity=connectivity, return_N=True)
    # If there is no lesion
    if N == 0:
        print('No lesion found after crop.')
        return mask, None, None
        
    # Get number of voxels, bounding box and centroid for each lesion
    stats = cc3d.statistics(output)

    # Number of voxels per lesion cluster with segid as key
    lesions = dict(enumerate(stats['voxel_counts']))
    # print(lesions)
    # First element is always the background
    del lesions[0]

    # Total number of lesion voxels
    lesion_total = np.sum(mask)
    # Initiate cumulated lesion volume
    volume = 0
    
    # print('LESION TOTAL :', lesion_total)
    c = 0
    for k in sorted(lesions, key=lesions.get, reverse=True):
        # If the biggest lesion is larger than 90% of the lesional volume
        if lesions[k] >= lesion_total*percentage and c == 0 and N > 1:
            # print(k, ': 1ST IF')
            smallest_lesion_size = lesions[k]
            lesion_cluster = np.where(output == k)
            new_mask[lesion_cluster] = 1
            lc +=1
            c = 1
        # print(k, lesions[k])
        volume += lesions[k]
        # Assure that we keep 90% of the biggest lesions by the end
        if volume <= lesion_total*percentage and N > 1: # MODIF FLORA FROM < TO <=
            # print(k, '2ND IF')
            lesion_cluster = np.where(output == k)
            new_mask[lesion_cluster] = 1
            lc += 1
            # print('keep lesion', k, volume)
            
    # Condition to keep if only 1 lesion
    if N == 1:
        # print(k, '3RD IF')
        lesion_cluster = np.where(output == 1)
        new_mask[lesion_cluster] = 1
        lc += 1
    
    if N > 1:
        smallest_lesion_size = lesion_cluster[0].shape[0]
    else: # Only 1 lesion
        smallest_lesion_size = lesion_total

    # print(lc, 'lesions kept over', N, 'in total (min lesion size =', smallest_lesion_size, 'voxels or', round(vox2mm3(smallest_lesion_size), 2), 'mm3)')

    return new_mask, smallest_lesion_size, round(vox2mm3(smallest_lesion_size), 2)



#######################
#   CROPPING VOLUMES  #
#######################

# Crop parameters to get a 256*360*220 sub-volume
# x_min , y_min, z_min, x_max, y_max, z_max = [180, 180, 200, 76, 180, 20]

# # Crop parameters to get a 128^3 sub-volume in the top-right part of the volume
x_min , y_min, z_min, x_max, y_max, z_max = [139, -61, 149, -11, 189, -21] 

# # Crop parameters to get a 128^3 sub-volume in the top-left part of the volume
# x_min , y_min, z_min, x_max, y_max, z_max = [150, 189, 139, -22, -61, -11]

print('Paramètres de crop :')
print('Axial', x_max +  x_min)
print('Sagittal',y_max +  y_min)
print('Coronal', z_max +  z_min)

# Write the choosen crop parameters in file
import os
os.remove("dataset/CropOption.txt")
f = open("dataset/CropOption.txt", "a")
for i in [x_min , y_min, z_min, x_max,y_max,z_max]:
    f.write(str(i)+'\n')
f.close()

def crop_center(img,cropx_i,cropy_i,cropz_i,cropx_e,cropy_e,cropz_e):
    x,y,z = img.shape[0],img.shape[1],img.shape[2]
    
    startx = x//2-(cropx_i)
    starty = y//2-(cropy_i)
    startz = z//2-(cropz_i)
    
    endx = x//2+(cropx_e)
    endy = y//2+(cropy_e)
    endz = z//2+(cropz_e)  
    return img[startx:endx,starty:endy,startz:endz]


######################
#    CONFIGURATION   #
######################

#axial
#sagittal
slices = 'coronal' # Annotation en coupe coronale # A SPECIFIER
#Don't resize if (0,0)
Resize_to = (0,0)#(256,256)# # A SPECIFIER
mask_dir = 'mask' # slices+'_masks_exp_L100_128' # A SPECIFIER
im_dir = 'img' # slices+'_images_exp_L100_128' # A SPECIFIER
path = 'dataset/TR-TL-128-grayscale-3S/'

LESION_PERCENTAGE = 1 # 0.9 # A SPECIFIER

PATLIST = [
    'Patient102_J54_50',
    'Patient110_J115_51',
    'Patient128_J41_72',
    'Patient129_J74_54',
    'Patient130_J29_47',
    'Patient131_J63_50',
    'Patient132_J37_55',
    'Patient17_J73_56',
    'Patient24_J42_67',
    'Patient25_J33_75',
    'Patient26_J36_70',
    'Patient34_J21_22',
    'Patient38_J31_44',
    'Patient39_J62_47',
    'Patient40_J62_45',
    'Patient41_J28_66',
    'Patient48_J12_43',
    'Patient51_J17_41',
    'Patient78_J20_65',
    'Patient79_J38_45',
    'Patient80_J27_57',
    'Patient81_J24_62',
    'Patient84_J90_29',
    'Patient88_J77_5',
    'Patient95_J46_45'
]

CONFIG = {
    'config0': ['Patient128_J41_72', 'Patient48_J12_43', 'Patient80_J27_57'],
    'config1': ['Patient25_J33_75', 'Patient26_J36_70', 'Patient34_J21_22'],
    'config2': ['Patient39_J62_47', 'Patient81_J24_62', 'Patient84_J90_29'],
    'config3': ['Patient129_J74_54', 'Patient40_J62_45', 'Patient79_J38_45'],
    'config4': ['Patient17_J73_56', 'Patient41_J28_66', 'Patient78_J20_65'],
    'config5': ['Patient24_J42_67', 'Patient38_J31_44'],
    'config6': ['Patient102_J54_50', 'Patient132_J37_55',],
    'config7': ['Patient131_J63_50', 'Patient88_J77_5'],
    'config8': ['Patient130_J29_47', 'Patient95_J46_45'],
    'config9': ['Patient110_J115_51', 'Patient51_J17_41']
}

# for config, pval in {'config5': ['Patient24_J42_67', 'Patient38_J31_44']}.items():
for config, pval in CONFIG.items():
    # print(config, pval)
    TEST_PATLIST = pval
    TRAIN_PATLIST = list(set(PATLIST) - set(TEST_PATLIST))
    print('\n[INFO]', config)
    # print(f'\n{len(TRAIN_PATLIST)} patients dans train : {TRAIN_PATLIST}')
    print(f'{len(TEST_PATLIST)} patients dans test : {TEST_PATLIST}\n')

    # Takes a volumes GT and Images and store the coronal slices in files for each patient
    # Read file to get crop parameters

    f = open("dataset/CropOption.txt", "r")
    crop = f.readlines()
    x_min , y_min, z_min, x_max,y_max,z_max = (int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3]), int(crop[4]), int(crop[5]))
    f.close()

    print('Paramètres de crop :')
    print('Axial', x_max +  x_min)
    print('Sagittal',y_max +  y_min)
    print('Coronal', z_max +  z_min)


    ##########################
    #   DATA PREPROCESSING   #
    ##########################


    def get_pat(file):
        pat = ('').join([file.split('\\')[1].split('_')[0].split('-')[0], file.split('\\')[1].split('_')[0].split('-')[1]])
        exam = file.split('\\')[1].split('_')[2].split('-')[0] + file.split('\\')[1].split('_')[2].split('-')[1] + '_' + file.split('\\')[1].split('_')[2].split('-')[3]
        return ('_').join([pat, exam])


    LabelPath = []
    ImagePath = []
    Patient_name = []


    n = 0
    for file in glob.glob("dataset/IMAGES/*.mat"): # A SPECIFIER
        ImagePath.append(file)
        # Patient_name.append(get_pat(file))
        # print(file.split('m')[1][1:-1])
        
    for file in glob.glob("dataset/GT/*.mat"): # A SPECIFIER
        Patient_name.append(get_pat(file))
        LabelPath.append(file)
        
    # print(x_min, y_min, z_min, x_max, y_max, z_max)

    # Create directories for train and test
    if not os.path.isdir(os.path.join(path, 'train-'+config)):
        os.makedirs(os.path.join(path, 'train-'+config, im_dir))
        os.makedirs(os.path.join(path, 'train-'+config, mask_dir))

    if not os.path.isdir(os.path.join(path, 'test-'+config)):
        os.makedirs(os.path.join(path, 'test-'+config, im_dir))
        os.makedirs(os.path.join(path, 'test-'+config, mask_dir))

    size_vox_min = []
    size_mm_min = []

    cl_tot = 0
    cl_train = 0
    cl_test = 0

    for data_idx in range(len(LabelPath)):
        # print('\n'+Patient_name[data_idx])
        # print(ImagePath[data_idx])
        # print(LabelPath[data_idx])
        # print(Patient_name[data_idx])
        
        # Load volumes
        image = hdf5storage.loadmat(ImagePath[data_idx])['data_repcom']
        mask = hdf5storage.loadmat(LabelPath[data_idx])['mapcont_repcom']
        
        # UNCOMMENT BELOW IF DENOISED VOLUMES
        # mask = hdf5storage.loadmat(LabelPath[data_idx])['mapcont_repcom']
        # try:
        #     image = hdf5storage.loadmat(ImagePath[data_idx])['data_repcom_dnsd']
        # except KeyError as e:
        #     # print(e)
        #     image = hdf5storage.loadmat(ImagePath[data_idx])['data_3D_dnsd']

        # Add padding before crop
        image = np.pad(image, ((100, 100), (100, 100), (100, 100)), 'constant')
        mask = np.pad(mask, ((100, 100), (100, 100), (100, 100)), 'constant')
        
        # Crop volume and mask to get the same shapes
        image = crop_center(image,x_min, y_min, z_min, x_max, y_max, z_max )
        mask = crop_center(mask,x_min, y_min, z_min, x_max, y_max, z_max )
        
        # Remove small lesions
        # mask, size_vox, size_mm = remove_small_lesions(mask, LESION_PERCENTAGE) # 1 = on conserve TOUTES LES LESIONS

        # size_vox_min.append(size_vox)
        # size_mm_min.append(size_mm)
        
        # Select the projection
        if( slices == 'coronal'):
            number_of_elements = mask.shape[2]
        elif( slices == 'sagittal'):
            number_of_elements = mask.shape[1]
        else:
            number_of_elements = mask.shape[0]

        cl = 0
        for i in range(number_of_elements):
            if( slices == 'coronal'):
                if i != 0 and i != (number_of_elements-1):
                    element_mask = mask[:,:,i] 
                    element_image = image[:,:,i-1:i+2] # 3 consecutive slices
                    # print(i, element_image.shape, element_mask.shape)
                elif i == 0:
                    element_mask = mask[:,:,i] 
                    element_image = image[:,:,:i+3] # 3 consecutive slices
                    # print(i, element_image.shape, element_mask.shape)
                else:
                    element_mask = mask[:,:,i] # 3 consecutive slices
                    element_image = image[:,:,i-2:]
                    # print(i, element_image.shape, element_mask.shape)
                    
            elif( slices == 'sagittal'):
                element_mask = mask[:,i,:]
                element_image = image[:,i,:]
            else:
                element_mask = mask[i,:,:]
                element_image = image[i,:,:]
                
            try:
                if Resize_to != (0,0): 
                    element_mask = element_mask.astype(np.float32)
                    element_image = element_image.astype(np.float32)
                    element_mask = cv2.resize(element_mask, Resize_to, interpolation = cv2.INTER_AREA)
                    element_image = cv2.resize(element_image, Resize_to, interpolation = cv2.INTER_AREA)
            
                if(np.max(element_mask) > 0): # Only keep slice with segmentation
                    cl += 1
                    # print(path+mask_dir+'/'+Patient_name[data_idx]+'/'+str(i)+'.png')
                    if Patient_name[data_idx] in TRAIN_PATLIST:
                        savepath = path + 'train-'+config+'/'
                        cl_train += 1
                    else:
                        savepath = path + 'test-'+config+'/'
                        cl_test += 1

                    # print(f'TRANCHE {i} : {np.min(element_image), np.max(element_image)} and {np.min(element_image.astype(np.uint8)), np.max(element_image.astype(np.uint8))}')
                    cv2.imwrite(savepath + mask_dir + '/' + Patient_name[data_idx] + '_' + str(i) + '_TL.png', element_mask.astype(np.uint8))
                    cv2.imwrite(savepath + im_dir + '/' + Patient_name[data_idx] + '_' + str(i) + '_TL.png', element_image.astype(np.uint8))
        
            except: # ValueError: zero-size array to reduction operation maximum which has no identity
                #error: OpenCV(4.5.4) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\resize.cpp:4051: error: (-215:Assertion failed) !ssize.empty() in function 'cv::resize'
                print('[ERROR]', Patient_name[data_idx])
                continue

            n = n + 1
                
            # update_progress(i / number_of_elements, Patient_name[data_idx] )
        # print(cl, 'tranches contenant des PWML sur', number_of_elements, 'au total.') 

        cl_tot += cl

        # update_progress(1,  Patient_name[data_idx])

    print(f"{cl_train} dans le jeu d'entraînement et {cl_test} dans celui de test.")

print(f'\n{cl_tot} tranches contenant des PWML au total.')