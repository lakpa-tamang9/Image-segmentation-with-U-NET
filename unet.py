import  tensorflow as tf
import os
import random
import numpy as np
import glob
from tqdm import tqdm
from PIL import Image
import cv2

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt


H = #specify your image height
W = #specify your image width
CH = #specify your number of image channels

X_TRAIN_PATH = #train path for images
Y_TRAIN_PATH = #train path for masks
TEST_PATH = #test path

#Reading folder with glob
x_train_ids = glob.glob(X_TRAIN_PATH + '/*')
print(len(x_train_ids))

y_train_ids = glob.glob(Y_TRAIN_PATH + '/*')
print(len(y_train_ids))

test_ids = glob.glob(TEST_PATH + '/*')
print(len(test_ids))



X_train = np.zeros((len(x_train_ids), H, W, CH), dtype = np.uint8)
print(X_train.shape)
Y_train = np.zeros((len(x_train_ids), H, W, 1), dtype = np.bool)
print(Y_train.shape)

for n, filename in tqdm(enumerate(x_train_ids)):
    img = imread(filename)
    img = resize(img, (H, W, CH), mode='constant', preserve_range=True)
    X_train[n] = img

    mask = np.zeros((H, W, 1), dtype=np.bool)
    for mask_file in (glob.glob(Y_TRAIN_PATH + '/*')):
        mask_ = imread(mask_file)
        mask_ = resize(mask_, (H, W, 1), mode='constant', preserve_range=True)
        mask = np.maximum(mask, mask_)

    Y_train[n] = mask

#test_images
X_test = np.zeros((len(test_ids), H, W, CH), dtype=np.uint8)
sizes_test = []
print('Resizing test images')
for n, filename in tqdm(enumerate(test_ids)):
    print(n)
    img = imread(filename)
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (H, W, CH), mode = 'constant', preserve_range=True)
    X_test[n] = img

print('Done!!!')

image_x = random.randint(0, len(x_train_ids))
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()

#Building U-Net
inputs = tf.keras.layers.Input((H, W, CH))
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

#Encoder structure of U-Net

c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
pool_1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
pool_2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
pool_3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
pool_4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Decoder structure of U-Net

u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding = 'same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding= 'same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding = 'same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding= 'same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding = 'same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding= 'same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding = 'same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding= 'same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

#Model checkpoint

chkp = tf.keras.callbacks.ModelCheckpoint('trained_model.h5', verbose=1, save_best_only=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir = 'logs')]

results = model.fit(X_train, Y_train, validation_split= 0.1, batch_size = 16, epochs = 25, callbacks=callbacks)

########################################

idx = random.randint(0, len(X_train))

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
print(preds_train)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
print(preds_val)
preds_test = model.predict(X_test, verbose=1)
print(preds_test)

preds_train_t = (preds_train > 0.8).astype(np.uint8)
print(preds_train_t)
preds_val_t = (preds_val > 0.8).astype(np.uint8)
print(preds_val_t)
preds_test_t = (preds_test > 0.8).astype(np.uint8)
print(preds_test_t)


#performing sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix],)
plt.show()
imshow(np.squeeze(Y_train[ix]),)
plt.show()
imshow(np.squeeze(preds_train_t[ix]),)
plt.show()

#Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()





