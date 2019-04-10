import random
import glob
import shutil
import csv

import cv2
import numpy as np

'''
Author: Garett MacGowan
This script takes image data and adds noise so that the CNN is forced
to better learn the essence of our symbols.
'''

def main(dataDirectory, newDataDirectory, noiseLimit, newInstancesPerImage):
  # Deleting the newData directory if it already exists
  print('Deleting old newData directory, if it exists')
  shutil.rmtree(newDataDirectory, True)
  # Copying the data directory to the newData directory
  print('Copying data into newData directory')
  shutil.copytree(dataDirectory, newDataDirectory)
  dataDir = glob.glob(newDataDirectory + '/*/*/')
  for index, directory in enumerate(dataDir):
    print('Progress: %' + str((index/len(dataDir))*100))
    fileNames = glob.glob(directory + '*.ppm')
    for img in fileNames:
      # Collecting original image data
      image = cv2.imread(img)
      labelCsvFile = glob.glob(directory + '*.csv')
      rowTemplate = []
      with open(labelCsvFile[0], 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)
        rowTemplate = next(reader)
      with open(labelCsvFile[0], 'a', newline='') as f:
        writer = csv.writer(f)
        # Applying a noising strategy to the image 10 times
        for index in range(newInstancesPerImage):
          newImage = addNoise(np.copy(image), noiseLimit)
          newImageName = img[:-4] + 'new' + str(index) + '.ppm'
          # Saving the new image with noise
          cv2.imwrite(newImageName, newImage)
          row = rowTemplate.copy()
          # NOTE that only the first and last column in the label will be correct.
          row[0] = newImageName[-19:]
          # Writing the label to the csv
          writer.writerow(row)

def addNoise(image, noiseLimit):
  imageVolume = image.shape[0]*image.shape[1]
  noiseVolumeLimit = imageVolume*noiseLimit
  # Create at least one instance of noise
  for _ in range(0, random.randint(7, 12)):
    xStart, xEnd, yStart, yEnd = generateNoiseBounds(image, noiseVolumeLimit)
    image[xStart : xEnd, yStart : yEnd, 0] = random.randint(0,255)
    image[xStart : xEnd, yStart : yEnd, 1] = random.randint(0,255)
    image[xStart : xEnd, yStart : yEnd, 2] = random.randint(0,255)
  return image

def generateNoiseBounds(image, noiseVolumeLimit):
  xEnd = random.randint(0, image.shape[0]-1)
  xStart = random.randint(0, xEnd)
  yEnd = random.randint(0, image.shape[1]-1)
  yStart = random.randint(0, yEnd)
  noiseVolume = abs(xStart - xEnd) * abs(yStart - yEnd)
  while (noiseVolume > noiseVolumeLimit):
    xEnd = random.randint(0, image.shape[0]-1)
    xStart = random.randint(0, xEnd)
    yEnd = random.randint(0, image.shape[1]-1)
    yStart = random.randint(0, yEnd)
    noiseVolume = abs(xStart - xEnd) * abs(yStart - yEnd)
  return xStart, xEnd, yStart, yEnd

'''
Parameters
  dataDirectory
  newDataDirectory
  Float noiseLimit
    Should be a number between 0 and 1 (inclusive of 1)
    which defines the % max volume of a component of noise, relative
    to the image size.
  Int newInstancesPerImage
    Should be a number >= 1.
    It defines the number of new images it will create from each image
    in the dataset, all containing some noise.
'''
main('./data', './newData', 0.07, 10)