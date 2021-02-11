# GBRAS_SW a software for steganalysis in the spatial domain.
GBRAS_SW is software for the detection of steganographic images in the spatial domain. An in-depth explanation of GBRAS_SW can be found in [1]. GBRAS_SW is state-of-the-art software for predicting steganographic images. GBRAS_SW has 30 SRM filters and a 3-fold TanH activation function for the preprocessing stage of steganographic images in the spatial domain. GBRAS_SW uses the ELU activation function in all feature extraction convolutions. GBRAS_SW uses shortcuts for feature extraction and separable and in-depth convolutions. GBRAS-Net does not use fully connected layers; the network uses a softmax directly after global average pooling.
## Prerequisites
The GBRAS_SW requires the following libraries and frameworks.

- Tensorflow 
-	numpy 
- opencv 
- argparse
- glob

The GBRAS_SW software was developed in the Python3 (3.8) programming language.
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

