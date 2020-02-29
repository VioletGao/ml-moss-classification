import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import re, cv2
import numpy
import glob
import os
dir = '/Users/violet/Documents/18WI/CSC499/evaluation/output/2255/'
ext = ".bmp" # whatever extension you want

pathname = os.path.join(dir, "*" + ext)


height = 1200
width = 1600
output = numpy.zeros((height,width,3))
horizontal = [800, 400, 200, 100, 50]
vertical = [600, 300, 150, 75, 38]

for img in glob.glob(pathname):
    index = re.findall(r'\d+', img)
    
    x = 0
    y = 0

    v = 4 # get rid of the number in dir
    while (v < len(index)):
        if (index[v] == '02'):
            y += vertical[(v-3)/2]
        v += 2

    ht = 5
    while (ht < len(index)):
        if (index[ht] == '02'):
            x += horizontal[(ht-4)/2]
        ht += 2
              
    image = cv2.imread(img)
    h = numpy.size(image, 0)
    w = numpy.size(image, 1)

    y2 = y+h
    x2= x+w
    #print(index)
    #print(y, y2, x, x2)
    output[y:y2,x:x2] = image

cv2.imwrite("IMG_2255.bmp", output)
print(dir[-5:], "done")
