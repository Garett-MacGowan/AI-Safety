import cv2
import glob

'''
Author: Garett MacGowan
This script is used to see the images in a given dataset. Press any key
to advance to the next image.
'''

def main(dataDirectory, newDataDirectory, dataToVisualize, datasetToUse):
  globObj = []
  if (dataToVisualize and datasetToUse):
    globObj = glob.glob(newDataDirectory + '/Training/*/*.ppm')
  if (not dataToVisualize and datasetToUse):
    globObj = glob.glob(newDataDirectory + '/Testing/*/*.ppm')
  if (dataToVisualize and not datasetToUse):
    globObj = glob.glob(dataDirectory + '/Training/*/*.ppm')
  if (not dataToVisualize and not datasetToUse):
    globObj = glob.glob(dataDirectory + '/Testing/*/*.ppm')
  for filename in globObj:
    image = cv2.imread(filename)
    cv2.imshow('Displaying training data', image)
    print('class ID ', filename)
    cv2.waitKey(0)

'''
Parameters
  dataDirectory
  newDataDirectory
  Boolean dataToVisualize
    True for training data
    False for testing data 
  Boolean datasetToUse
    False for old data
    True for new data
'''
main('./data', './newData', True, True)
