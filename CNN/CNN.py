import numpy as np

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras import optimizers,losses
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
import keras

import cv2
import os

batch_size = 10
num_epochs = 5
num_datapoints = 1

input_base_directory = "../Datasets/Images/"

test_pos_directory = "../datasets/testset/Face/"
test_neg_directory = "../datasets/testset/Other/"
pos_test_filename_Format = test_pos_directory + "Face_frame_"
neg_test_filename_Format = test_neg_directory + "Other_frame_"


output_dim = 2

def str_to_img(s):
	img = list(map(lambda x: 1 if x=='1' else 0,s.split(",")))
	return img

def shape(line):
	return np.array(list(map(lambda x: int(x), line.split(",")))).reshape(-1,28)

def make_model(alpha):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape=(28,28,1)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))
	# model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	# model.add(Dropout(0.5))
	model.add(Dense(output_dim, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
	optimizer=keras.optimizers.Adadelta(),
	metrics=['accuracy'])
	
	return model

def get_file_data(filename_x,filename_y,one_hot_labels=True):
	with open(filename_x) as file_x,open(filename_y) as file_y:
		r_labels = []; r_data = [];
		for line in file_x:
			r_data.append(shape(line))
		if one_hot_labels:
			for line in file_y:
				r_labels.append(labels[int(line.strip())])
			r_labels=np.array(r_labels)
		else:
			for line in file_y:
				r_labels.append(int(line.strip()))
	r_data = np.array(r_data).reshape(len(r_data),28,28,1)

	return r_data,r_labels

def get_test_accuracy(test_data,test_labels,model):
	predicted_labels = model.predict(test_data)
	predicted_labels= list(map(lambda x: np.argmax(x),predicted_labels))
	num_right = 0
	for predicted,actual in zip(predicted_labels,test_labels):
		num_right += 1 if predicted == actual else 0
	return num_right/len(test_labels)

def get_images(num_datapoints,sub_dir):
	img_base_path = input_base_directory + sub_dir + "/"
	images = np.array([])
	for i in range(1,num_datapoints+1):
		img_path = img_base_path + sub_dir + "_frame_" + str(i).zfill(2) + ".png"
		img = cv2.imread(img_path)
		print(img.shape)
		np.append(images,img)


def main():
	test_data = np.array([])
	test_labels = np.array([])

	np.append(test_data,get_images(num_datapoints,"Face"))
	np.append(test_labels, ([1,0] for i in range(num_datapoints)))

	# test_data.append(get_images(num_datapoints,"Other"))
	# test_labels.append([0,1] for i in range(num_datapoints))

	




if __name__ == '__main__':
	main()