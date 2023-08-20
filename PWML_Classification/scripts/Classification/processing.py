import numpy as np
import cc3d

#######################################
#    CONNECTED COMPONENT EXTRACTION   #
#######################################


def extract_vignette(image, centroid, dimensions=(64, 64)):
    """
    Extract a vignette of given dimensions from an image
    centered around the centroid coordinates (connected component)
    """
    # print(centroid[1], centroid[0])
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
    coordinates = [[x1, x2], [y1, y2]]
    # plt.imshow(vignette, 'gray')
    # plt.show()

    assert vignette.shape == dimensions

    return vignette #, coordinates



########################
#    POST PROCESSING   #
########################


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
        return mask#, 0
        
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

        volume += lesions[k]
        
        # Assure that we keep 90% of the biggest lesions by the end
        if volume <= lesion_total*percentage and N > 1:
            lesion_cluster = np.where(output == k)
            new_mask[lesion_cluster] = 1
            lc += 1
            
    # Condition to keep if only 1 lesion
    if N == 1:
        lesion_cluster = np.where(output == 1)
        new_mask[lesion_cluster] = 1
        lc += 1
    
    if N > 1:
        smallest_lesion_size = lesion_cluster[0].shape[0]
    else: # Only 1 lesion
        smallest_lesion_size = lesion_total

    # print(lc, 'lesions kept over', N, 'in total (min lesion size =', smallest_lesion_size, 'voxels or', round(vox2mm3(smallest_lesion_size), 2), 'mm3)')

    return new_mask #, smallest_lesion_size