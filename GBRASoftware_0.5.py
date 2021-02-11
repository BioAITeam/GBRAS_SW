#GBRASoftware, can predict if a folder with images is COVER or STEGO

#sudo -s
#cd /media/ia/Datos1/DocReinel/Steganalysis/H_Brayan_A_Arteaga/Frankenstein_Project/GBRASoftware
#conda activate tf22
#python GBRASoftware_0.5.py -i ./images -m ./models/S-UNIWARD_0.4bpp.hdf5


#Libraries
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import cv2
import argparse
import glob

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
def cover_or_stego(model, image):
    image = np.reshape(image,(1,256,256,1)) 
    acc_val = model.predict(image)
    prediction = np.round(acc_val) 
    if (prediction == np.array([1, 0])).all():
        print("Prediction: COVER, accuracy: {:.1f}%\n".format(acc_val[0][0]*100)) 
    else:
        print("Prediction: STEGO, accuracy: {:.1f}%\n".format(acc_val[0][1]*100))

#Load a model
##Option models ./models/
###S-UNIWARD_0.4bpp.hdf5, S-UNIWARD_0.2bpp.hdf5, WOW_0.4bpp.hdf5, WOW_0.2bpp.hdf5
def load_model(path_model):
    model = tf.keras.models.load_model(path_model, custom_objects={'Tanh3':Tanh3}) 
    return model

if __name__ == '__main__':
    print("\n############################################################")
    print("#                                                          #")
    print("#  GBRAS-Net: A Convolutional Neural Network Architecture  #")
    print("#              for Spatial Image Steganalysis              #")
    print("#                                                          #")
    print("############################################################\n")

    ### read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path-images', required=True, dest='path_image',help='Path of the images to be classified')
    parser.add_argument('-m', '--path-model', required=True, dest='path_model',help='Path of the model')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v1.0')

    options = parser.parse_args()
    path_image = options.path_image
    path_model = options.path_model

    Loaded_model = load_model(path_model)
    # to load all images in the path
    for image in glob.glob(path_image+"/*.pgm"):
        Loaded_image = load_image(image)
        cover_or_stego(Loaded_model, Loaded_image) #Prediction