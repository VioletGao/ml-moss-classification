import image_slicer
import csv
import os
from glob import glob

###for fn in glob('/Users/violet/Documents/18WI/CSC499/mix_images/*.bmp'):
###    image_slicer.slice(fn, 4)
##
### open the file in universal line ending mode 
##with open(file_path + 'x16x16_pred_mix.csv', 'rU') as infile:
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
### extract the variables you want
##names = data['Slice']


''' convert image index to the actual filename of the image '''
def process_filename(index):
  if len(index) > 4:
    pos = list(split_by_n(index[4:],2))
    name = "IMG_" + str(index) + "_".join(pos) + ".png"
  else:
    name = "IMG_" + str(index) + ".bmp"

  #print(name)
  return name
  
  

def slice_image(indices, iteration):
  
  read_dir = '/Users/violet/Documents/18WI/CSC499/images/' + str(iteration) + '/'
  save_dir = '/Users/violet/Documents/18WI/CSC499/images/' + str(iteration*4) + '/'

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
  for index in indices:
    filename = process_filename(index)
    fn = read_dir + filename
    tiles = image_slicer.slice(fn, 4, save=False)
    image_slicer.save_tiles(tiles, directory = save_dir, prefix=filename[:-4])

  return save_dir

