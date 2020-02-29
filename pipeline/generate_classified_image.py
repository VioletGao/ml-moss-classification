import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
from glob import glob
import os
import cv2
import csv
import numpy as np
import re

##
##path = '/Users/violet/Documents/18WI/CSC499/'
##image_path = '/Users/violet/Documents/18WI/CSC499/images/x16x16x4/'
##
##
### open the file in universal line ending mode 
##with open(path + 'evaluation/x16x16x4_pred_mix.csv', 'rU') as infile:
##  # read the file as a dictionary for each row ({header : value})
##  reader = csv.DictReader(infile)
##  data = {}
##  for row in reader:
##    for header, value in row.items():
##      try:
##        data[header].append(value)
##      except KeyError:
##        data[header] = [value]
##
##names = data['Slice']
##
##
##    for fn in names:
##        name = fn.rsplit('/', 1)[-1][:-4]
##        img = cv2.imread(image_path + fn, 0)
##        height = img.shape[0]
##        width = img.shape[1]
##        #print(height, width)
##        image = create_blank(width, height, rgb_color=color)


def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


def generate_image(images, state, iteration):

    indices = set(images)

    img_dir = '/Users/violet/Documents/18WI/CSC499/images/' + str(iteration) + '/'
    file_dir = '/Users/violet/Documents/18WI/CSC499/evaluation/output/'
    ext = ".bmp"    
    yellow = (244, 212, 66)
    green = (146, 244, 65)
    grey = (211,211,211)

    if state == 'dry':
        color = yellow
    elif state == 'wet':
        color = green

    for fn in glob(img_dir + '*' + ext):
        filename = fn.rsplit('/', 1)[-1]
        index = re.sub(r'[a-zA-Z_.]', '', filename)
        if index in indices:
            img = cv2.imread(fn, 0)
            height = img.shape[0]
            width = img.shape[1]
            new_img = create_blank(width, height, rgb_color=color)            
            cv2.imwrite(file_dir + filename , new_img)

