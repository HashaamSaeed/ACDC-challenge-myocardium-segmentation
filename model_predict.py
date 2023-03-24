
import os
import random
import numpy as np
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import sys
import glob
import cv2
import tensorflow as tf
import scipy.misc
import matplotlib
import time
import re
import argparse
import nibabel as nib
from medpy.metric.binary import hd, dc
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from textwrap import wrap

###################################################################################################

#np.set_printoptions(threshold=sys.maxsize)

model = tf.keras.models.load_model("unet_model")


## seed remains same for rand operator otherwise everytime it runs we will get diff val for all val with rand
seed = 42
np.random.seed = seed

###################################################################################################
# Path definitions

train_image_dir = 'ACDC_Dataset_PNG_192x192/train'   
test_image_dir = 'ACDC_Dataset_PNG_192x192/test'

img_fname       = 'images'  # folder_name train images
mask_fname      = 'masks'  # folder_name of train masks

def get_train_imgs():
    img_path = os.path.join(train_image_dir,img_fname)
    images = glob.glob(os.path.join(img_path,'*.*'))
    mask_path = os.path.join(train_image_dir,mask_fname)
    masks = glob.glob(os.path.join(mask_path,'*.*'))
    return [os.path.basename(image) for image in images],[os.path.basename(mask) for mask in masks]

# print(get_tain_imgs())

def get_test_imgs():
	test_img_path = os.path.join(test_image_dir,img_fname)
	test_img = glob.glob(os.path.join(test_img_path,'*.*'))
	return[os.path.basename(testimage) for testimage in test_img],[]


TRAIN_IMGS = get_train_imgs()
TEST_IMGS = get_test_imgs()

all_batches = TRAIN_IMGS
all_test = TEST_IMGS
# print(all_test)


img_path  = os.path.join(train_image_dir,img_fname)
mask_path = os.path.join(train_image_dir,mask_fname)
test_img_path = os.path.join(test_image_dir,img_fname)
#print('img_path',img_path)

###################################################################################################

# Some important parameter to change prediction image in the dataset and threshold probability for 
# the predicted image

ix=9       ## Just change number here in the range of 0-1611 to predict the segmentation of the training image
predprob = 0.01  ## this is the probability threshold for the predicted image pixels 

###################################################################################################

# loading data to be predicted
# loading train images and some path manipulation to load pixdim
def imgload(img_path,all_batches,ix):
	X_train = np.zeros((1,192,192,1),dtype=np.uint8) # unsigned 8 didgit
	img1 = os.path.join(img_path,all_batches[0][ix])
	print('all_batches',all_batches[0][ix])
	imgix=os.path.splitext(img1)[0]
	imgix=os.path.splitext(imgix)[0]
	var1 = imgix
	#print('var1',var1)
	var2 = ".nii.gz"
	var3 = f"{var1}{var2}"
	var3=os.path.relpath(var3, 'ACDC_Dataset_PNG_192x192/train/images')
	#print('var3',var3)
	var4=var3 
	var5="ACDC_Dataset_Seperated_data_seperated/train_set/"
	var5=f"{var5}{var4}"
	#print('var5',var5)
	c_img  = cv2.imread(img1,0)
	c_img  = np.expand_dims(c_img,axis=-1)
	c_img  = np.expand_dims(c_img,axis=0)
	X_train = c_img
	return X_train,var3,var5


imgpredict=imgload(img_path,all_batches,ix)
	
# loading train mask images	
def maskload(mask_path,all_batches,ix):
	Y_train = np.zeros((1,192,192,1), dtype=np.bool)  ## dtype is boolean cuz its the mask
	#mask = np.zeros((192,192, 1), dtype=np.bool)
	img2 = os.path.join(mask_path,all_batches[1][ix])
	mask = cv2.imread(img2,0)
	mask = np.expand_dims(mask, axis=-1)
	Y_train = mask
	return Y_train

maskpredict=maskload(mask_path,all_batches,ix)


'''
# loading test images
for test in range(len(all_test[0])):
	img3 = os.path.join(test_img_path,all_test[0][test])
	test_img1  = cv2.imread(img3,0)
	test_img1  = np.expand_dims(test_img1,axis=-1)
	X_test[test] = test_img1
'''


# print(Y_train)
#print('img',X_train.shape,'mask',Y_train.shape,'test_img',X_test.shape)
# print(Y_train.dtype)

###################################################################################################
# loading model and predicting the LV in the image

def UNet_predict(imgpredict):
	preds_train = model.predict(imgpredict[0], verbose=1)## every pixel here has a probability value as the output from the U-net from 0-1
	#preds_test = model.predict((X_test[:int(X_test.shape[ix])], verbose=1), verbose=1)
	return preds_train

preds_train=UNet_predict(imgpredict)


def processpredict(preds_train):
	preds_train_t=np.zeros((1,192,192,1),dtype=np.uint8)
	preds_train_t = (preds_train > predprob).astype(np.uint8)  ## this tensor has the image in binary form since it accepts pixle that has a probability greater than 0.5
	#preds_test_t = (preds_test > 0.6).astype(np.uint8)
	#print('preds_train_t',preds_train_t.shape)
	pred_img= np.squeeze(preds_train_t)*255
	#print('preds_img squeeze',pred_img.shape)
	return preds_train_t, pred_img


processpredict1=processpredict(preds_train)

#maskhere = np.zeros((192,192), dtype=np.bool)
#maskhere= pred_img
#cv2.imwrite('hello.png',pred_img)
Y_trainmask= np.squeeze(maskpredict)
#print('Y_train squeeze',Y_trainmask.shape)

###################################################################################################
# Now we calculate and define the metrics like volume and dice

def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.header


def metrics(img_gt, img_pred, voxel_size):
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    img_pred: np.array
    Array of the predicted segmentation map.

    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.

    Return
    ------
    A list of metrics in this order, [Dice LV, Volume LV, Err LV(ml)]
    """

    if img_gt.ndim  != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    res = []
    # Loop on each classes of the input images
    for c in [1]:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        #gt_c_i[gt_c_i != c] = 0

        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        #pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)

        # Compute the Dice
        dice = dc(gt_c_i, pred_c_i)

        # Compute volume
        volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
        volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.

        res += [dice, volpred, volgt]

    return res

imghere=load_nii(imgpredict[2])
header1= imghere
header1=header1['pixdim']
print('header file for pixdim',header1)
#print('voxel size',header1[2])
#print('slice num',header1[3])
voxel1=header1[1]
voxel2=header1[2]
slicenum=int(header1[3])

header= metrics(Y_trainmask,processpredict1[1],[voxel1,voxel2,slicenum])

#print('metrics here',metrics(Y_trainmask,processpredict1[1],[voxel1,voxel2,slicenum]))

###################################################################################################
# Calculates area within the segmented mask

def threshold(pred_img):
	#raw = imread('hello.png', as_gray=True)
	threshold = threshold_otsu(pred_img)
	thresholded = pred_img > threshold
	# Label by default assumes that zeros correspond to "background".
	# However, we actually want the background pixels in the center of the ring,
	# so we have to "disable" that feature.
	labeled = label(thresholded, background=2)
	overlay = label2rgb(labeled,bg_label=0)

	#fig, axes = plt.subplots(1, 3)
	#axes[3].imshow(pred_img, cmap='gray')
	#axes[4].imshow(thresholded, cmap='gray')
	'''
	if cycle==1:
		axes[3].imshow(overlay)
		axes[3].set_title('ED OTSU Pred Mask')
	else:
		axes[7].imshow(overlay)
		axes[7].set_title('ES OTSU Pred Mask')
'''
	

	convex_areas = []
	areas = []
	for properties in regionprops(labeled):
	    areas.append(properties.area)
	    convex_areas.append(properties.convex_area)

	# take the area with the smallest convex_area
	idx = np.argmin(convex_areas)
	area_of_interest = areas[idx]
	print(f"My area of interest has {area_of_interest} pixels.")

	return area_of_interest


var6=os.path.splitext(imgpredict[1])[0]	
var6=os.path.splitext(var6)[0]
###################################################################################################
#post processing for the predicted mask 

def postmorph(foreground,background):
	foreground1 = foreground
	#foreground= np.expand_dims(foreground,axis=2)
	background1 = np.squeeze(background)
	foreout = np.zeros((192,192,3),dtype=np.uint8)
	kernel = np.ones((2,2),np.uint8)
	gradient = cv2.morphologyEx(foreground1, cv2.MORPH_GRADIENT, kernel)
	

	for k in range (3):
		for i in range(192):
			for j in range(192):
				foreout[i,j,k]=background1[i,j]
				if k==1 & gradient.item(i,j)>0:
					foreout[i,j,k]=gradient[i,j]


	return foreout



###################################################################################################

# Plotting results for the image for ED or ES

if '_frame01' in var6:
	numpixel= threshold(processpredict1[1])
	dicecoeff=int(header[0]*100)
	# First method by the lyon lab
	Vol1=header[1]
	# My method of calculating the said metrics
	Vol2=(((numpixel)*(voxel1)*(voxel2)*(slicenum))/1000)
	foreout= postmorph(processpredict1[1],imgpredict[0])

	fig, axes = plt.subplots(1, 6)
	axes[0].imshow(np.squeeze(imgpredict[0]),cmap='gray')
	axes[0].set_title('ED Slice')
	axes[1].imshow(np.squeeze(maskpredict),cmap='gray')
	axes[1].set_title('ED Mask')
	axes[2].imshow(foreout)
	axes[2].set_title('ED Predicted Segementation')
	#axes[2].imshow(np.squeeze(processpredict1[0]))
	#axes[2].set_title('ED Prediced Mask')

else:
	numpixel1= threshold(processpredict1[1])
	dicecoeff1=int(header[0]*100)
	# First method by the lyon lab
	Vol11=header[1]
	# My method of calculating the said metrics
	Vol22=(((numpixel1)*(voxel1)*(voxel2)*(slicenum))/1000)
	foreout1= postmorph(processpredict1[1],imgpredict[0])
	
	fig, axes = plt.subplots(1, 6)
	axes[0].imshow(np.squeeze(imgpredict[0]),cmap='gray')
	axes[0].set_title('ES Slice')
	axes[1].imshow(np.squeeze(maskpredict),cmap='gray')
	axes[1].set_title('ES Mask')
	axes[2].imshow(foreout1)
	axes[2].set_title('ES Predicted Segementation')
	#axes[2].imshow(np.squeeze(processpredict1[0]))
	#axes[2].set_title('ES Prediced Mask')




###################################################################################################
## Calculating ED and ES volumes and EF percentage for the patient with the complementing ED or ES slice

if '_frame01' in var6:
	newix=ix+slicenum
	print('newix',newix)
	imgpredict1=imgload(img_path,all_batches,newix)
	maskpredict=maskload(mask_path,all_batches,newix)
	imghere=load_nii(imgpredict1[2])
	header1= imghere
	header1=header1['pixdim']
	voxel1=header1[1]
	voxel2=header1[2]
	slicenum=header1[3]
	preds_train1=UNet_predict(imgpredict1)
	processpredict1=processpredict(preds_train1)
	Y_trainmask= np.squeeze(maskpredict)
	header= metrics(Y_trainmask,processpredict1[1],[voxel1,voxel2,slicenum])
	dicecoeff1=int(header[0]*100)
	foreout1= postmorph(processpredict1[1],imgpredict1[0])
	#fig, axes = plt.subplots(1, 4)
	axes[3].imshow(np.squeeze(imgpredict1[0]),cmap='gray')
	axes[3].set_title('ES Slice')
	axes[4].imshow(np.squeeze(maskpredict),cmap='gray')
	axes[4].set_title('ES Mask')
	axes[5].imshow(foreout1)
	axes[5].set_title('ES Predicted Segementation')
	#axes[5].imshow(np.squeeze(processpredict1[0]))
	#axes[5].set_title('ES Prediced Mask')
	numpixel1= threshold(processpredict1[1])
	Vol11=header[1]
	Vol22=(((numpixel1)*(voxel1)*(voxel2)*(slicenum))/1000)

else:
	newix=ix-slicenum
	imgpredict1=imgload(img_path,all_batches,newix)
	maskpredict=maskload(mask_path,all_batches,newix)
	imghere=load_nii(imgpredict1[2])
	header1= imghere
	header1=header1['pixdim']
	voxel1=header1[1]
	voxel2=header1[2]
	slicenum=header1[3]
	preds_train1=UNet_predict(imgpredict1)
	processpredict1=processpredict(preds_train1)
	Y_trainmask= np.squeeze(maskpredict)
	header= metrics(Y_trainmask,processpredict1[1],[voxel1,voxel2,slicenum])
	dicecoeff=int(header[0]*100)
	foreout= postmorph(processpredict1[1],imgpredict1[0])
	#fig, axes = plt.subplots(1, 4)
	axes[3].imshow(np.squeeze(imgpredict1[0]),cmap='gray')
	axes[3].set_title('ED Slice')
	axes[4].imshow(np.squeeze(maskpredict),cmap='gray')
	axes[4].set_title('ED Mask')
	axes[5].imshow(foreout)
	axes[5].set_title('ED Predicted Segementation')
	#axes[5].imshow(np.squeeze(processpredict1[0]))
	#axes[5].set_title('ED Prediced Mask')
	numpixel= threshold(processpredict1[1])
	Vol1=header[1]
	Vol2=(((numpixel)*(voxel1)*(voxel2)*(slicenum))/1000)
	


EF1=((Vol1-Vol11)/Vol1)*100

EF2=((Vol2-Vol22)/Vol2)*100

#print('EF1',EF1,'EF2',EF2)

###################################################################################################

fig.suptitle("\n".join(wrap(f"For {var6} we will look at the pair ED and ES slice Segementation and metrics, No. of pixels in the LV for ED is {numpixel} pixels and ES is {numpixel1} pixels ,\
the dice coefficeint for prediction of ED is {dicecoeff}% and for ES is {dicecoeff1}%,\
the volume of the predicted segmentation in LV for a single slice in ED is {format(Vol2,'.2f')}mL and for a single slice in \
ES is {format(Vol22,'.2f')}mL, and the resulting Ejection fraction is {format(EF2,'.2f')}%", 120)),fontsize=20,fontweight='bold')


'''
fig.suptitle("\n".join(wrap(f"For {var6} we will look at the pair ED and ES slice Segementation and metrics, No. of pixels in the LV for ED is {numpixel} pixels and ES is {numpixel1} pixels ,\
the dice coefficeint for prediction of ED is {dicecoeff}% and for ES is {dicecoeff1}%,\
the volume of the predicted segmentation in LV for a single slice in ED is {format(Vol1,'.2f')}mL, for a single slice in \
ES is {format(Vol11,'.2f')}mL, and the resulting Ejection fraction is {format(EF1,'.2f')}% (This method of volume calc is proposed by lab in Lyon) according to a method i found online \
the volume of the predicted segmentation in LV for a single slice in ED is {format(Vol2,'.2f')}mL and for a single slice in \
ES is {format(Vol22,'.2f')}mL, and the resulting Ejection fraction is {format(EF2,'.2f')}%", 120)),fontsize=20,fontweight='bold')
'''

plt.show()

###################################################################################################
