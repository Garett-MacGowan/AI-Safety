import tensorflow as tf
import numpy as np
import glob
import csv
import cv2
import time

from sklearn.utils import shuffle
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import load_model, save_model
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping

'''
Author: Garett MacGowan
This python script creates, trains, and tests a model given a specified
dataset. Specifically, it allows one to test the difference between the
performance of the raw dataset and the dataset with noisy images added.
'''

def main(dataDirectory, newDataDirectory, datasetToUse, loadModel):
  print('in main')
  # If the dataset is the new dataset
  print('Collecting data... This may take some time.')
  nTrainingData, nTrainingLabels, nTestingData, nTestingLabels = readData(newDataDirectory)
  # If the dataset is the original dataset
  oTrainingData, oTrainingLabels, oTestingData, oTestingLabels = readData(dataDirectory)
  # Randomize training data
  oTrainingData, oTrainingLabels = shuffle(oTrainingData, oTrainingLabels, random_state=0)
  nTrainingData, nTrainingLabels = shuffle(nTrainingData, nTrainingLabels, random_state=0)
  earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto', restore_best_weights=True)
  # Creating model
  if (loadModel and datasetToUse):
    model = load_model('./models/newDataModel.h5')
  elif (loadModel and not datasetToUse):
    model = load_model('./models/ogDataModel.h5')
  else:
    if (datasetToUse):
      print('Training on new data')
      currentTime = time.time()
      modelName = f"Street-Sign-Safety-CNN-Noised-Data-{currentTime}"
      tensorboard = TensorBoard(log_dir=f"./logs/{modelName}")
      model = createModel(nTrainingData[0].shape)
      model.fit(x=nTrainingData, y=nTrainingLabels, epochs=50, validation_split=0.2, callbacks=[tensorboard, earlyStopping])
    else:
      print('Training on old data')
      currentTime = time.time()
      modelName = f"Street-Sign-Safety-CNN-{currentTime}"
      tensorboard = TensorBoard(log_dir=f"./logs/{modelName}")
      model = createModel(oTrainingData[0].shape)
      model.fit(x=oTrainingData, y=oTrainingLabels, epochs=50, validation_split=0.2, callbacks=[tensorboard, earlyStopping])
  if (not loadModel and datasetToUse):
    model.save('./models/newDataModel.h5')
  elif (not loadModel and not datasetToUse):
    model.save('./models/ogDataModel.h5')

  print('Evaluating on original dataset')
  testLoss, testAccuracy = model.evaluate(oTestingData, oTestingLabels)
  print('Test accuracy: ', testAccuracy)
  print('Test loss: ', testLoss)

  print('Evaluating on noised dataset')
  testLoss, testAccuracy = model.evaluate(nTestingData, nTestingLabels)
  print('Test accuracy: ', testAccuracy)
  print('Test loss: ', testLoss)

def readData(dataDirectory):
  dataDir = glob.glob(f"{dataDirectory}/*/")
  trainingData = []
  trainingLabels = []
  testingData = []
  testingLabels = []
  for item in dataDir:
    if ('Training' in item):
      trainingData, trainingLabels = collectImagesAndLabels(item, trainingData, trainingLabels)
    if ('Testing' in item):
      testingData, testingLabels = collectImagesAndLabels(item, testingData, testingLabels)
  return trainingData, trainingLabels, testingData, testingLabels

def collectImagesAndLabels(item, data, labels):
  classPaths = glob.glob(f"{item}*/")
  for path in classPaths:
    labelCsv = glob.glob(f"{path}*.csv")
    classLabel = 0
    with open(labelCsv[0], 'r') as f:
      try:
        reader = csv.reader(f, delimiter=';')
        next(reader)
        # Finding correct class label
        classLabel = next(reader)[-1]
      except:
        '''
        Will occur when there is no data for this class. Infering class label, but not really
        necessary.
        '''
        classLabel = str(int(labels[-1]) + 1)
    imagePaths = glob.glob(f"{path}*.ppm")
    for img in imagePaths:
      image = cv2.imread(img)
      # Resizing image so that it can be passed into CNN
      image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
      data.append(image)
      labels.append(classLabel)
  return np.array(data), np.array(labels)

def createModel(inputShape):
  model = tf.keras.Sequential()

  model.add(layers.Conv2D(8, kernel_size=(3,3), activation='relu', padding='same', input_shape=(128,128,3)))
  model.add(layers.BatchNormalization())
  model.add(layers.Conv2D(8, kernel_size=(3,3), activation='relu', padding='same'))
  model.add(layers.BatchNormalization())

  model.add(layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
  model.add(layers.BatchNormalization())

  model.add(layers.MaxPool2D(pool_size=(2,2), strides=2, padding='valid'))

  model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(128,128,3)))
  model.add(layers.BatchNormalization())
  model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
  model.add(layers.BatchNormalization())

  model.add(layers.MaxPool2D(pool_size=(2,2), strides=2, padding='valid'))

  model.add(layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
  model.add(layers.BatchNormalization())

  model.add(layers.MaxPool2D(pool_size=(2,2), strides=2, padding='valid'))
  # Flattening the nodes so that the dense layer can be applied.
  model.add(layers.Flatten())
  model.add(layers.Dense(62, activation='sigmoid'))
  # Final layer is dense layer with 62 nodes, since there are 62 classes.
  model.add(layers.Dense(62, activation='softmax'))
  '''
  Compiling the model with the adam optimizer. Using sparse categorical crossentropy
  because class labels are not one-hot encoded.
  '''
  model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
  )
  return model

'''
Parameters
  dataDirectory
  newDataDirectory
  Boolean datasetToUse
    False for old data
    True for new data
  Boolean loadModel
    True to load the saved model
    False to re-make (re-train) and save the model
'''
main('./data', './newData', False, True)