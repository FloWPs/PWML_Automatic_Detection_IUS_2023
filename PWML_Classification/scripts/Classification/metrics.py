#######################
#  IMPORT LIBRAIRIES  #
#######################

import time as time
import numpy as np
import cv2
import cc3d

import tensorflow as tf
from keras import backend as K

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix


########################
#    CUSTOM METRICS    #
########################


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def MCC(y_true, y_pred):
    labels = tf.argmax(input=y_true, axis=1)
    preds = tf.argmax(input=y_pred, axis=1)
    return matthews_corrcoef(labels, preds)


############################
#   EVALUATION FUNCTIONS   #
############################

def recall_and_precision_lesion(y_true, y_pred, lesion_min=10, boundig_box_margin=15):

    recall = []
    precision = []

    for s in range(y_pred.shape[2]):
        # Select slice
        # print(s)
        # element_image = volume[:, :, s]
        element_mask = y_true[:, :, s].astype(np.uint8)
        element_prediction = y_pred[:, :, s].astype(np.uint8)

        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(image=element_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        # number of true lesions in the groundtruth mask
        nb_lesions = len(contours)

        # initialize a list to store TRUE lesion detection and compute recall & precision afterwards
        lesions_detected = np.zeros(nb_lesions)
        # number of false positives
        FP = 0

        # apply connected component analysis to the thresholded image
        connectivity = 8
        output = cv2.connectedComponentsWithStats(element_prediction, connectivity, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        # loop over the number of unique connected component labels, skipping
        # over the first label (as label zero is the background)
        for i in range(1, numLabels):
            # extract the connected component statistics for the current# label
            # x = stats[i, cv2.CC_STAT_LEFT]
            # y = stats[i, cv2.CC_STAT_TOP]
            # w = stats[i, cv2.CC_STAT_WIDTH]
            # h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            # print('\nConnected component', i, ':', area, 'pixels')
            # print('Barycenter coordinates :', (cX, cY))

            # If there is no lesion and the cc predicted is significative
            if len(contours)==0:
                # print('NO LESIONS')
                keepArea = area > lesion_min
                if keepArea:
                    FP += 1

            else:
                # Check if the current CC is a TRUE POSITIVE
                cc = 0
                
                for j in range(len(contours)):
                    # ensure the width, height, and area are all neither too small
                    # nor too big
                    # keepWidth = w > 5 and w < 50
                    # keepHeight = h > 45 and h < 65
                    keepArea = area > lesion_min
                    # ensure the connected component we are
                    # examining passes all three tests
                    if keepArea:
                        # construct a mask for the current connected component and
                        # then take the bitwise OR with the mask
                        # print("[INFO] keeping connected component '{}'".format(i))
                        # print("[CONTOUR "+str(j)+"]")

                        # Check if barycenter belong to the polygonal contour of TRUE lesion
                        result = cv2.pointPolygonTest(contours[j], (cX, cY), False)
                        # print(result)
                        # Check if barycenter belong to the rectangular contour of TRUE lesion
                        x,y,w,h = cv2.boundingRect(contours[j]) # BOUNDING BOX
                        x -= boundig_box_margin
                        y -= boundig_box_margin
                        w += boundig_box_margin
                        h += boundig_box_margin
                        res = x <= cX < x+w and y <= cY < y + h
                        # print(res)
                        if res:
                            lesions_detected[j] = 1
                            cc = 1
                    else:
                        # print("[INFO] removing connected component '{}'".format(i))
                        # we ignore the lesion for computing recall and precision
                        cc = 1

                # If cc DO NOT belong to any of the TRUE lesions
                if cc == 0:
                    FP += 1

        # if no lesions in GT AND correct prediction (no lesion in the predicted mask)
        # print(nb_lesions, FP, recall, precision)
        if nb_lesions==0:
            if FP == 0:
                precision.append(1)
            if FP > 0: # FP > 0
                precision.append(0)
        
        else:
            recall.append(np.sum(lesions_detected)/len(lesions_detected))
            if (np.sum(lesions_detected) + FP) > 0:
                precision.append(np.sum(lesions_detected)/(np.sum(lesions_detected) + FP))
            else:
                precision.append(0)

    # print(len(recall), recall)
    # print(len(precision), precision)
    # print('Rappel (Lesion-wise) =', np.mean(recall))
    # print('Precision (Lesion-wise) =', np.mean(precision))
    f1score = 2 * (np.mean(recall) * np.mean(precision)) / (np.mean(recall) + np.mean(precision))
    return np.mean(recall), np.mean(precision), f1score


def dice_coef2(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    intersection = np.sum(y_true_f * y_pred_f)
    
    return (2. * intersection + smooth) / (union + smooth)
        

def confusion_matrix_slice(y_true, y_pred):
    y_true_f = y_true
    y_pred_f = y_pred
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    FN_pred = []
    TP_pred = []

    for i in range(y_pred_f.shape[2]):
        cm_pw = confusion_matrix(y_true_f[:, :, i].flatten(), y_pred_f[:, :, i].flatten())
        if np.max(y_pred_f[:, :, i]) == 1 and np.max(y_true_f[:, :, i]) == 1:
            # We check if the prediction intersect with the groundtruth
            if cm_pw[1][1] > 0:
                TP += 1
                TP_pred.append(i)
            else:
                FN += 1
                FN_pred.append(i)
            # TP += 1
        elif np.max(y_pred_f[:, :, i]) == 1 and np.max(y_true_f[:, :, i]) == 0:
            FP += 1
        elif np.max(y_pred_f[:, :, i]) == 0 and np.max(y_true_f[:, :, i]) == 0:
            TN += 1
        else:
            FN += 1

    return TP, FP, TN, FN, FN_pred, TP_pred


def get_cv_scores(confusion_matrix, decimal=4):
    """
    Return classification scores from confusion matrix.
    """
    TN = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TP = confusion_matrix[1][1]

    accuracy = np.round((TP + TN) / (TP + FP + TN + FN), decimal)
    recall = np.round(TP / (TP + FN), decimal)
    precision = np.round(TP / (TP + FP), decimal)
    mcc = np.round(((TP * TN) - (FP * FN)) / ((TP + FP) * (TP + FN) * (TN + FP ) * (TN + FN))**(1/2), decimal)

    return {'Accuracy': accuracy, 'Recall': recall, 'Precision': precision, 'MCC': mcc}


def get_scores(patient, threshold, train_dim, y_true, y_pred, decimal=4):
    scores = {'PatientID': patient, 'Threshold': threshold, 'Training Dim': train_dim}

    # # IoU
    # scores['IoU_b'] = np.round(jaccard_score(y_true.flatten(), y_pred.flatten(), average='binary'), decimal)
    # scores['IoU_m'] = np.round(jaccard_score(y_true.flatten(), y_pred.flatten(), average='macro'), decimal)
    # scores['IoU_w'] = np.round(jaccard_score(y_true.flatten(), y_pred.flatten(), average='weighted'), decimal)

    # Dice
    scores['Dice'] = np.round(dice_coef2(y_true, y_pred), decimal)

    # Recall & Precision (Lesion-wise)
    recall, precision, f1 = recall_and_precision_lesion(y_true, y_pred, lesion_min=10, boundig_box_margin=15)
    scores['Recall (Lesion-wise)'] = np.round(recall, decimal)
    scores['Precision (Lesion-wise)'] = np.round(precision, decimal)
    scores['F1-Score (Lesion-wise)'] = np.round(f1, decimal)

    # Confusion matrix (pixel-wise)
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    # try:
    #     scores['True Intersection (Pixel-wise)'] = np.round(cm[1] / np.sum(cm[1]), decimal)[1]
    # except RuntimeWarning:
    if cm.shape != (1, 1):
        scores['True Intersection (Pixel-wise)'] = np.round(cm[1] / np.sum(cm[1]), decimal)[1]
        scores['FN (Pixel-wise)'] = cm[1][0]
        scores['TP (Pixel-wise)'] = cm[1][1]
    else:
        scores['True Intersection (Pixel-wise)'] = None
        scores['FN (Pixel-wise)'] = None
        scores['TP (Pixel-wise)'] = None

    # Confusion Matrix (slice-wise)
    TP, FP, TN, FN, FN_pred, TP_pred = confusion_matrix_slice(y_true, y_pred)

    try:
        # MCC
        scores['MCC (Slice-wise)'] = np.round((TP*TN-FP*FN) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)), decimal)
        # Recall
        scores['Recall (Slice-wise)'] = np.round(TP/(TP+FN), decimal)

        # Precision
        scores['Precision (Slice-wise)'] = np.round(TP/(TP+FP), decimal)

        # F1-Score
        scores['F1-Score (Slice-wise)'] = np.round(TP / (TP + (1/2) * (FP + FN)), decimal)

        # Accuracy
        scores['Accuracy (Slice-wise)'] = np.round((TP+TN)/(TP+FP+FN+TN), decimal)

    except ZeroDivisionError:
        scores['MCC (Slice-wise)'] = None
        scores['Recall (Slice-wise)'] = None
        scores['Accuracy (Slice-wise)'] = None
        scores['F1-Score (Slice-wise)'] = None
        scores['Dice'] = None

    scores['TN'] = TN
    scores['FP'] = FP
    scores['FN'] = FN
    scores['TP'] = TP
    scores['FN_pred'] = FN_pred
    scores['TP_pred'] = TP_pred

    return scores