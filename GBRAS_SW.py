#GBRAS_SW



#Libraries
import tensorflow as tf
import numpy as np
import cv2
import argparse
import glob
import xlsxwriter
import os
import datetime

#3xTanH ACTIVATION FUNCTION
def Tanh3(x):
    tanh3 = tf.keras.activations.tanh(x)*3
    return tanh3

#Load an image
def load_image(path_image):
	I = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
	I = np.array(I)
	return I

#Prediction if an image is cover or stego
def cover_or_stego(model, image):
    image = np.reshape(image,(1,256,256,1))
    acc_val = model.predict(image)
    prediction = np.round(acc_val)
    if (prediction == np.array([1, 0])).all():
        worksheet.write(row, 1, "cover")
        worksheet.write(row, 2, float("{:.1f}".format(acc_val[0][0]*100))) 
        print("Prediction: COVER, accuracy: {:.1f}%\n".format(acc_val[0][0]*100))
    else:
        worksheet.write(row, 1, "stego")
        worksheet.write(row, 2, float("{:.1f}".format(acc_val[0][1]*100))) 
        print("Prediction: STEGO, accuracy: {:.1f}%\n".format(acc_val[0][1]*100))

#Load a model
##Option models ./models/
###S-UNIWARD_0.4bpp.hdf5, S-UNIWARD_0.2bpp.hdf5, WOW_0.4bpp.hdf5, WOW_0.2bpp.hdf5
def load_model(path_model):
    global model_name
    model_name = os.path.basename(path_model)
    if model_name.endswith('.hdf5'):
        model_name = model_name[:-5]
    
    model = tf.keras.models.load_model(path_model, custom_objects={'Tanh3':Tanh3})
    return model

def write_exel_file():
    global worksheet,workbook,row
    path_log = './logs'
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    workbook = xlsxwriter.Workbook(path_log+'/GBRAS-Net_'+str(datetime.datetime.now().isoformat()[:19].replace("T", "_").replace(":","-"))+"_"+model_name+".xlsx") 
    worksheet = workbook.add_worksheet()
    cell_format = workbook.add_format({'bold': True, 'font_color': 'red'})
    worksheet.write(0, 0, "image", cell_format) 
    worksheet.write(0, 1, "prediction", cell_format) 
    worksheet.write(0, 2, "accuracy [%]", cell_format) 
    row = 1

if __name__ == '__main__':
    print("\n############################################################")
    print("#                                                          #")
    print("#  GBRAS-Net: A Convolutional Neural Network Architecture  #")
    print("#              for Spatial Image Steganalysis              #")
    print("#                                                          #")
    print("#                    Trained models:                       #")
    print("#                 S-UNIWARD_0.4bpp.hdf5                    #")
    print("#                 S-UNIWARD_0.2bpp.hdf5                    #")
    print("#                    WOW_0.4bpp.hdf5                       #")
    print("#                    WOW_0.2bpp.hdf5                       #")
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
    write_exel_file()

    # to load all images in the path
    for image in glob.glob(path_image+"/*.pgm"):
        Loaded_image = load_image(image)
        cover_or_stego(Loaded_model, Loaded_image) #Prediction
        worksheet.write(row, 0, os.path.basename(image)) 
        row += 1
    workbook.close() 
