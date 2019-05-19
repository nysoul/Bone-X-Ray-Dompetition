import os
import numpy as np
import cv2
import random
import keras.backend as K

def load_path(root_path = '../valid/XR_ELBOW', size = 512):
	'''
	load MURA data


	'''

	Path = []
	labels = []
	for root,dirs,files in os.walk(root_path): 
		for name in files:
			path_1 = os.path.join(root,name)
			Path.append(path_1)
			if root.split('_')[-1]=='positive':	 
				labels+=[1]   	          	 
			else:
			    labels+=[0]
	print (len(Path))
	labels = np.asarray(labels)
	return Path, labels

def load_image(Path = '../valid/XR_ELBOW', size = 512):
	Images = []
	for path in Path:
		image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		image = cv2.resize(image,(size,size))
		#image = randome_rotation_flip(image,size)
		Images.append(image)

	Images = np.asarray(Images).astype('float32')

	mean = np.mean(Images[:, :, :])			#normalization
	std = np.std(Images[:, :, :])
	Images[:, :, :] = (Images[:, :, :] - mean) / std
	
	if K.image_data_format() == "channels_first":
		Images = np.expand_dims(Images,axis=1)		   
	if K.image_data_format() == "channels_last":
		Images = np.expand_dims(Images,axis=3)              
	return Images




if __name__ == '__main__':
	load_path()
