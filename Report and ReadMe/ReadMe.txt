********************************
Author: Garett MacGowan
Student Number: 10197107
Email: 15gm1@queensu.ca
********************************

This document describes execution instructions for running
the python scripts in this project.

********************************
Python Version Verified Compatibility: 3.6.8
Python Library Requirements:
- TensorFlow 1.13.x
    • pip install tensorflow
    or
    • pip tensorflow-gpu
        • Tensorflow-gpu requires CUDA 10 and CUDNN.
	  See https://www.tensorflow.org/install/gpu for instruction
    • You must use TensorFlow 1.13.x because it contains keras packages
      with specific functions that are used.
- NumPy
    • pip install numpy
- OpenCV
    • pip install opencv-python
- Scikit-learn
    • pip install scikit-learn
********************************

********************************
Original Data
- The original data is available at https://btsd.ethz.ch/shareddata/
  next to the BelgiumTS for Classification (cropped images): section.
  There are two links, training and testing.
********************************

How To Use:

IMPORTANT
For the model to run correctly, you need to run dataNoising.py first so that the new noised
dataset can be generated.

***
dataNoising.py
***
In its default state, dataNoising.py will execute and create data with instances of noise,
each component taking up as much as 7% of the total volume of the original image. It will
create 10 new images with noise for every image in the original dataset. Use this function
if you would like to generate the new dataset.

***
Main.py
***
In its default state, main.py will execute an existing model trained on the old data.
Parameters for the script can be found at the bottom of the file. All parameters
are described there. You can change the parameters to re-train the models and/or
select the model to train (train on original un-noised data or new noise-added data).
This script will output the accuracy results of the model you have run to console, so
make sure you are running the script in a terminal, or in an interpreter.

***
inspectImages.py
***
In its default state, inspectImages.py will show you the training data for the new dataset
(the existing noised dataset, or the dataset you have generated with dataNoising.py)