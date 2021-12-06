# GBRAS_SW: a software for steganalysis in the spatial domain.
GBRAS_SW is a software for the detection of steganographic images in the spatial domain. It is based in the GBRAS architecture that is in-depth explanate in [1]. GBRAS_SW shows high precision rates for the prediction of steganographic images. This software maintains, for preprocessing stage, the 30 SRM filters and has a 3xTanH activation function. Also, it uses the ELU activation function in all feature extraction convolutions, and shortcuts for feature extraction and separable and depthwise convolutions. This software does not use fully connected layers; the network uses a softmax directly after global average pooling.
## Prerequisites
The GBRAS_SW requires the following libraries and frameworks.

- TensorFlow 
-	numPy 
- OpenCV 
- argparse
- glob
- XlsxWriter
- os
- datetime


GBRAS_SW  was developed in the Python3 (3.8) programming language.

## Installation:
We highly recommend to use and install Python packages within an Anaconda enviroment. To create, execute the command below:
```
conda create --name GBRAS_SW python=3.8
```
So, activate it
```
conda activate GBRAS_SW 
```
installed the framework
```
conda install -c anaconda keras-gpu==2.4.3
```
Now, install the libraries.
```
pip install opencv-python
pip install scikit-image
conda install -c conda-forge argparse
conda install -c conda-forge xlsxwriter
conda install -c jmcmurray os
conda install -c trentonoliphant datetime
```
## GBRAS_SW execution
After installing all the prerequisites, you must clone the repository of the current version of GBRAS_SW using.
```
git clone https://github.com/BioAITeam/GBRAS_SW.git
```
Then you might run as following:
```
python GBRAS_SW.py -i ./images -m ./models/S-UNIWARD_0.4bpp.hdf5
```
In the repository, there are two folders, one with images and the other with models. The images folder contains eighty cover and stego images for testing the software. You can add more images to the folder to test the software's accuracy in detecting cover and stego image in the spatial domain.  The format of the images is Portable Gray Map (PGM). In the model folder, there are four models S_UNIWARD and WOW, with two payloads, 0.4 and 0.2 bpp, respectively. You can choose any of the four models to perform a cover or stego image prediction, for example:

```
python GBRAS_SW.py -i ./images -m ./models/WOW_0.4bpp.hdf5
```
## Authors
Universidad Autónoma de Manizales (https://www.autonoma.edu.co/)

- Reinel Tabares Soto
- Harold Brayan Arteaga Arteaga
- Mario Alejandro Bravo Ortiz
- Alejandro Mora Rubio
- Daniel Arias Grazon
- Jesus Alejandro Alzate Grisales
- Simon Orozco Arias

Universidad de Caldas (http://ucaldas.edu.co/)

- Gustavo Isaza

Universidad de Antioquia (http://udea.edu.co/)

- Raul Ramos Pollan

## Citing

If you use our project for your research or if you find this paper and repository helpful, please consider citing the work:

T. -S. Reinel et al., "GBRAS-Net: A Convolutional Neural Network Architecture for Spatial Image Steganalysis," in IEEE Access, vol. 9, pp. 14340-14350, 2021, doi: [10.1109/ACCESS.2021.3052494](https://doi.org/10.1109/ACCESS.2021.3052494). 

```
@ARTICLE{GBRAS2021,  
  author={Reinel, Tabares-Soto and Brayan, Arteaga-Arteaga Harold and Alejandro, Bravo-Ortiz Mario and Alejandro, Mora-Rubio and Daniel, Arias-Garzón and Alejandro, Alzate-Grisales Jesús and Buenaventura, Burbano-Jacome Alejandro and Simon, Orozco-Arias and Gustavo, Isaza and Raúl, Ramos-Pollán},  
  journal={IEEE Access},   
  title={GBRAS-Net: A Convolutional Neural Network Architecture for Spatial Image Steganalysis},   
  year={2021},  
  volume={9},  
  number={},  
  pages={14340-14350},  
  doi={10.1109/ACCESS.2021.3052494}
}
```

This paper was published as a journal paper in IEEE Access. ([PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9328287))
