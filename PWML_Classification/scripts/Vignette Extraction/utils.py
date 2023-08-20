import cc3d
import numpy as np
import tensorflow as tf
import glob, os
import hdf5storage


def crop_center(img, cropx_i, cropy_i, cropz_i, cropx_e, cropy_e, cropz_e):
    x, y, z = img.shape[0], img.shape[1], img.shape[2]

    startx = x // 2 - (cropx_i)
    starty = y // 2 - (cropy_i)
    startz = z // 2 - (cropz_i)

    endx = x // 2 + (cropx_e)
    endy = y // 2 + (cropy_e)
    endz = z // 2 + (cropz_e)
    return img[startx:endx, starty:endy, startz:endz]


def vox2mm3(vol):
    """
    Convert number of voxels to volume in mm^3
    """
    voxel_volume = 0.168**3
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
        return mask#, None, None
        
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
    
    c = 0
    for k in sorted(lesions, key=lesions.get, reverse=True):
        # If the biggest lesion is larger than 90% of the lesional volume
        if lesions[k] >= lesion_total*percentage and c == 0 and N > 1:
            smallest_lesion_size = lesions[k]
            lesion_cluster = np.where(output == k)
            new_mask[lesion_cluster] = 1
            lc +=1
            c = 1
        # print(k, lesions[k])
        volume += lesions[k]
        # Assure that we keep 90% of the biggest lesions by the end
        if volume <= lesion_total*percentage and N > 1:
            lesion_cluster = np.where(output == k)
            new_mask[lesion_cluster] = 1
            lc += 1
            # print('keep lesion', k, volume)
            
    # Condition to keep if only 1 lesion
    if N == 1:
        lesion_cluster = np.where(output == 1)
        new_mask[lesion_cluster] = 1
        lc += 1
    
    if N > 1:
        smallest_lesion_size = lesion_cluster[0].shape[0]
    else: # Only 1 lesion
        smallest_lesion_size = lesion_total

    print(lc, 'lesions kept over', N, 'in total (min lesion size =', smallest_lesion_size, 'voxels or', round(vox2mm3(smallest_lesion_size), 2), 'mm3)')

    return new_mask#, smallest_lesion_size, round(vox2mm3(smallest_lesion_size), 2)


def preprocessing_prediction(path, cropOption, mask=False, lesion_percentage=1): # MODIF FLORA

    if mask:
        image = hdf5storage.loadmat(path)['mapcont_repcom']
        
    else:
         image = hdf5storage.loadmat(path)['data_repcom']
        # try:
        #     image = hdf5storage.loadmat(path)['data_repcom_dnsd'] # (volume débruité)
        # except KeyError as e:
        #     # print(e)
        #     image = hdf5storage.loadmat(path)['data_3D_dnsd']

    Init_Volume = [image.shape[0], image.shape[1], image.shape[2]]
    x_min, y_min, z_min, x_max, y_max, z_max = cropOption
    image = np.pad(image, ((100, 100), (100, 100), (100, 100)), 'constant')
    reduced_image = crop_center(image, x_min, y_min,z_min, x_max, y_max, z_max)
    # print("\nImage preprocessed shape :", reduced_image.shape) 

    if mask:
        reduced_image = remove_small_lesions(reduced_image, lesion_percentage)
        # print('MASK.shape')
        # print(reduced_image.shape)
        reduced_image = tf.image.resize(tf.cast(reduced_image, dtype=tf.uint8), (128, 128))
        # print(reduced_image.shape)
    else:
        # print('IMAGE.shape')
        # print(reduced_image.shape)
        reduced_image = tf.image.resize(reduced_image, (128, 128))
        # print(reduced_image.shape)

    return reduced_image, Init_Volume