from typing import Any, Union
import os, sys
import configparser
import numpy as np
import configparser
from matplotlib import pyplot as plt
#Keras
from keras.models import model_from_json
from keras.models import Model
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score

from help_functions import *
from extract_patches import  *

config = configparser.RawConfigParser()
config.read('configuration.txt')
path_data = config.get('data paths', 'path_local')

test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
test_imgs_orig: object = load_hdf5(test_imgs_original)
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]
test_border_masks = path_data + config.get('data paths', 'test_border_masks')
test_border_masks = load_hdf5(test_border_masks)
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))

name_experiment = config.get('experiment name', 'name')
path_experiment = './' +name_experiment +'/'
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))

N_visual = int(config.get('testing settings', 'N_group_visual'))
average_mode: Union[object, Any] = config.getboolean('testing settings', 'average_mode')




patches_imgs_test = None
new_height = None
new_width = None
masks_test  = None
patches_masks_test = None
if average_mode == True:
    patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
        test_imgs_original = test_imgs_original,  #original
        test_groudTruth = path_data + config.get('data paths', 'test_groundTruth'), 
        Imgs_to_test = int(config.get('testing settings', 'full_images_to_test')),
        patch_height = patch_height,
        patch_width = patch_width,
        stride_height = stride_height,
        stride_width = stride_width,
    )
else:
    patches_imgs_test, patches_masks_test = get_data_testing(
        test_imgs_original = test_imgs_original, 
        test_groudTruth = path_data + config.get('data paths', 'test_groundTruth'),
        Imgs_to_test = int(config.get('testing settings', 'full_images_to_test')),
        patch_height = patch_height,
        patch_width = patch_width,
    )


best_last = config.get('testing settings', 'best_last')
model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')
predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)
print ("predicted images size :")
print (predictions.shape)
pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "original")
#pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "threshold")


pred_imgs = None
orig_imgs = None
gtruth_masks = None
if average_mode == True:
    pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
    orig_imgs = my_PreProc(test_imgs_orig[0:pred_imgs.shape[0],:,:,:])    #originals
    gtruth_masks = masks_test  #ground truth masks
else:
    pred_imgs = recompone(pred_patches,13,12)       # predictions
    orig_imgs = recompone(patches_imgs_test,13,12)  # originals
    gtruth_masks = recompone(patches_masks_test,13,12)  #masks
    
kill_border(pred_imgs, test_border_masks)
orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
gtruth_masks = gtruth_masks[:,:,0:full_img_height,0:full_img_width]

assert (orig_imgs.shape[0]==pred_imgs.shape[0] and orig_imgs.shape[0]==gtruth_masks.shape[0])
N_predicted = orig_imgs.shape[0]
group = N_visual
assert (N_predicted%group==0)
for i in range(int(N_predicted/group)):
    orig_stripe = group_images(orig_imgs[i*group:(i*group)+group,:,:,:],group)
    masks_stripe = group_images(gtruth_masks[i*group:(i*group)+group,:,:,:],group)
    pred_stripe = group_images(pred_imgs[i*group:(i*group)+group,:,:,:],group)
    total_img = np.concatenate((orig_stripe, pred_stripe), axis=0)
    visualize(total_img,path_experiment+name_experiment +"_Original_GroundTruth_Prediction"+str(i))#.show()
   
