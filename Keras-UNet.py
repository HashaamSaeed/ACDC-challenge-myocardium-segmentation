###################################################################################################
'''
Code by : Adem Saglam and Syed Muhammad Hashaam Saeed


'''
###################################################################################################


import tensorflow as tf
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

#np.set_printoptions(threshold=sys.maxsize)
## seed remains same for rand operator otherwise everytime it runs we will get diff val for all val with rand
seed = 42
np.random.seed = seed


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

X_train = np.zeros((len(all_batches[0]),192,192,1),dtype=np.uint8) # unsigned 8 didgit
Y_train = np.zeros((len(all_batches[1]),192,192,1), dtype=np.bool)  ## dtype is boolean cuz its the mask
X_test = np.zeros((len(all_test[0]),192,192,1), dtype=np.uint8)

# loading train images
for num in range(len(all_batches[0])):
	img1 = os.path.join(img_path,all_batches[0][num])
	c_img  = cv2.imread(img1,0)
	c_img  = np.expand_dims(c_img,axis=-1)
	X_train[num] = c_img
	
# loading train mask images	
mask = np.zeros((192,192, 1), dtype=np.bool)
for mask_file in range(len(all_batches[1])):
	img2 = os.path.join(mask_path,all_batches[1][mask_file])
	mask = cv2.imread(img2,0)
	mask = np.expand_dims(mask, axis=-1)
	Y_train[mask_file] = mask


# loading test images
for test in range(len(all_test[0])):
	img3 = os.path.join(test_img_path,all_test[0][test])
	test_img1  = cv2.imread(img3,0)
	test_img1  = np.expand_dims(test_img1,axis=-1)
	X_test[test] = test_img1


# print(Y_train)
print('img',X_train.shape,'mask',Y_train.shape,'test_img',X_test.shape)
# print(Y_train.dtype)


'''
# to display loaded images
image_x = random.randint(0,len(all_batches[0]))
imshow(np.squeeze(X_train[image_x]))
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()
image_y= random.randint(0,len(all_test[0]))
imshow(np.squeeze(X_test[image_y]))
plt.show()
'''



'''# for viewing IMG and Mask together
cv2.imshow('image',X_train[image_x])
imshow(np.squeeze(Y_train[image_x]))
plt.show()
cv2.waitKey(0)
'''






#Build the model
inputs = tf.keras.layers.Input((192,192,1))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)## converting image into float points by division

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
## 16 is the number of filters and 3x3 their size, kernel initialiser initialises whats inside comv kernal
## multiplied by s means what the layer is applied on
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
## accuracy metric to measure models performance after training
model.summary()

################################
#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('ACDC_data_model_trained.h5', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'), ## patience measn it'll do 3 more epochs after it stops changing loss or any val you choose
        tf.keras.callbacks.TensorBoard(log_dir='logs')]






results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=20, callbacks=callbacks)
## validation_split means percentage data for validation
## callbacks are basically what  needs to be saved as checkpoints
####################################

model.save('U-Net LV Segment model')
idx = random.randint(0, len(X_train))

## model.predict gives the prediction/output
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1) ## every pixel here has a probability value as the output from the U-net from 0-1
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.55).astype(np.uint8)  ## this tensor has the image in binary form since it accepts pixle that has a probability greater than 0.5
preds_val_t = (preds_val > 0.55).astype(np.uint8)
preds_test_t = (preds_test > 0.55).astype(np.uint8)


ix = random.randint(0, len(preds_train_t))

# Perform a sanity check on some random training samples and show them
# imshow(X_train[ix])
imshow(np.squeeze(X_train[ix]))
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(np.squeeze(X_train[int(X_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()


