from __future__ import print_function

import os
import numpy as np
import keras.backend as K
import cv2

def load_path(root_path , size = 512):
	
	Path = []
	
	for root,dirs,files in os.walk(root_path): 
		for name in files:
			path_1 = os.path.join(root,name)
			Path.append(path_1)
			
	return Path

def load_image(Path, size = 512):
	Images = []
	for path in Path:
		image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		image = cv2.resize(image,(size,size))
		Images.append(image)
        
	Images = np.asarray(Images).astype('float32')

	mean = np.mean(Images[:, :, :])			
	std = np.std(Images[:, :, :])
	Images[:, :, :] = (Images[:, :, :] - mean) / std
	
	if K.image_data_format() == "channels_first":
		Images = np.expand_dims(Images,axis=1)		   
	if K.image_data_format() == "channels_last":
		Images = np.expand_dims(Images,axis=3)             
	return Images



input_image_directory = 'predict'

from keras.models import load_model
print('Loading Model.. `')
model = load_model('save_models/best_MURA_modle.h5')
print('Sucessfully loaded model \n ')
im_size=320
X_valid_path= load_path(input_image_directory, size = im_size)  
X_valid = load_image(X_valid_path,im_size)

print(model.predict(X_valid))
    