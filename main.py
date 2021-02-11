#GBRASoftware, can predict if a image is COVER or STEGO

#sudo -s
#conda activate tf22
#cd /media/ia/Datos1/DocReinel/Steganalysis/H_Brayan_A_Arteaga/Frankenstein_Project/GBRASoftware
#python main.py

#Libraries
import tensorflow as tf
from keras import backend as K
import numpy as np
import cv2

#3xTanH ACTIVATION FUNCTION
def Tanh3(x):
    tanh3 = K.tanh(x)*3
    return tanh3

#Load a image of a path
def load_image(path_image):
	I = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE) 
	I = np.array(I)
	return I

#Prediction if a image is cover or stego
def cover_or_stego(image):
    image = np.reshape(image,(1,256,256,1))  
    prediction = np.round(model.predict(image))
    print("\n\n\nGBRAS-Net: A Convolutional Neural Network Architecture for Spatial Image Steganalysis \n")
    #print(prediction)
    if (prediction == np.array([1, 0])).all():
        print("Prediction: COVER\n")
    else:
        print("Prediction: STEGO\n")

#Load a model
##Option models ./models/
###S-UNIWARD_0.4bpp.hdf5, S-UNIWARD_0.2bpp.hdf5, WOW_0.4bpp.hdf5, WOW_0.2bpp.hdf5
def load_model(path_model):
	global model
	model = tf.keras.models.load_model(path_model, custom_objects={'Tanh3':Tanh3}) 



#Select the image path to predict
path_image_c = '/media/ia/Datos1/DocReinel/Steganalysis/H_Brayan_A_Arteaga/Frankenstein_Project/GBRAS-Net/DATABASES/BOSSbase-1.01/cover'
path_image_s = '/media/ia/Datos1/DocReinel/Steganalysis/H_Brayan_A_Arteaga/Frankenstein_Project/GBRAS-Net/DATABASES/BOSSbase-1.01/S-UNIWARD/0.4bpp/stego'

path_image = path_image_s+'/1.pgm'
path_model = './models/S-UNIWARD_0.4bpp.hdf5' #name of the selected model

load_model(path_model)
cover_or_stego(load_image(path_image)) #Prediction