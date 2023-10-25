# mssfnet
## Introduction
MSSFNet is a deep learning model designed for extracting Martian surface minerals from Compact Reconnaissance Imaging Spectrometer for Mars (CRISM) images. This repository contains the official implementation of MSSFNet along with training and evaluation scripts.

This repository offers a PyTorch-based implementation of MSSFNet, including training scripts, prediction scripts, and a comprehensive dataset for testing purposes.

## Experimental Data
The training dataset and the test dataset are in the 'data' folder. Please extract 'data1.rar' and 'data2.rar' into the 'data' folder.

## Requirements
h5py==2.8.0

matplotlib==2.2.3

numpy==1.15.1

scikit_learn==0.19.2

scipy==1.1.0

torch==2.0.1

torchvision==0.15.2

## Training
Run train.py to initiate the training process. Set the train_dataset variable to specify the training dataset.

## Evaluation
Run predict.py to perform testing.
