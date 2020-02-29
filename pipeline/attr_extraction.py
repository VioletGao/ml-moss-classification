##from ij import IJ, ImageJ, ImagePlus, ImageStack
##from ij.plugin import ChannelSplitter, ImageCalculator
##from ij.process import ImageProcessor, ImageConverter
##from ij.measure import ResultsTable, Measurements
##from ij.plugin.frame import RoiManager
##from ij.plugin.filter import ParticleAnalyzer
##from java.lang import Double
import imageJ
import csv
import glob

def extractAttributes(path):
	imp = IJ.openImage(path)
	#stack = imp.getImageStack()
	name = path.rsplit('/', 1)[-1]
	#imps = ChannelSplitter.split(imp)
	red = ChannelSplitter.getChannel(imp, 1)
	green = ChannelSplitter.getChannel(imp, 2)
	#imps[0].show() # Channel 1
	#imps[1].show() # Channel 2
	
	calc = ImageCalculator()
	red_channel = ImagePlus("red", red)
	green_channel = ImagePlus("green", green)
	#red_channel.show()
	#green_channel.show()
	
	subtraction = calc.run("subtract create", green_channel, red_channel)
	#subtraction.show()
	
	subtraction.getProcessor().setThreshold(9, 255, ImageProcessor.NO_LUT_UPDATE)
	IJ.run(subtraction, "Convert to Mask", "")
	IJ.run(subtraction, "Watershed", "")
	#subtraction.updateAndDraw()
	
	# Create a table to store the results
	table = ResultsTable()
	# Create a hidden ROI manager, to store a ROI for each blob or cell
	roim = RoiManager(True)
	# Create a ParticleAnalyzer, with arguments:
	# 1. options (could be SHOW_ROI_MASKS, SHOW_OUTLINES, SHOW_MASKS, SHOW_NONE, ADD_TO_MANAGER, and others; combined with bitwise-or)
	# 2. measurement options (see [http://imagej.net/developer/api/ij/measure/Measurements.html Measurements])
	# 3. a ResultsTable to store the measurements
	# 4. The minimum size of a particle to consider for measurement
	# 5. The maximum size (idem)
	# 6. The minimum circularity of a particle
	# 7. The maximum circularity
	pa = ParticleAnalyzer(ParticleAnalyzer.SHOW_OUTLINES, Measurements.AREA, table, 100, 3000)
	pa.setHideOutputImage(True)
	 
	pa.analyze(subtraction)
	#  print "Area ok"
	#else:
	#  print "There was a problem in analyzing", subtraction
	 
	# The measured areas are listed in the first column of the results table, as a float array:
	area = table.getColumn(0)
	if area is None:
	  length = float(0)
	  ave_area = 0
	else:
	  length = len(area)
	  ave_area = sum(area)/length

        imp_size = imp.width * imp.height
        num_of_par = length/imp_size  # standardized number of particles per image size
	
	table = ResultsTable()
	pa = ParticleAnalyzer(ParticleAnalyzer.SHOW_NONE, Measurements.SHAPE_DESCRIPTORS, table, 100, 3000)
	pa.setHideOutputImage(True)
	
	pa.analyze(subtraction)
	#  print "Shape ok"
	#else:
	#  print "There was a problem in analyzing", subtraction
	
	
	table.saveAs("/Users/violet/Desktop/eggs1.csv")
	table = ResultsTable().open2("/Users/violet/Desktop/eggs1.csv")
	
	circ = table.getColumn(0)
	AR = table.getColumn(1)
	if (length == 0):
	  ave_circ = 0
	  ave_AR = 0
	else:
  	  ave_circ = sum(circ)/length
	  ave_AR = sum(AR)/length
	
	
	
	hsb_convert = ImageConverter(imp).convertToHSB()
	hue = ChannelSplitter.getChannel(imp, 1)
	hue_channel = ImagePlus("hue", hue)
	#hue_channel.show()
	#hue_channel.setThreshold(63, 96, ImageProcessor.NO_LUT_UPDATE)
	#hue_channel.updateAndDraw()
	ip = hue_channel.getProcessor().convertToFloat()  
	pixels = ip.getPixels() 
	size = float(len(pixels))
	count = float(0)
	for p in pixels: 
	  if p >= 63 and p <= 96: 
	    count += 1
	greenness = count/size
  	
	return([name, num_of_par, ave_area, ave_circ, ave_AR, greenness, "w"])


     
def batch_processing(filepath, iteration):
  output_path = "/Users/violet/Documents/18WI/CSC499/evaluation/" + str(iteration) + ".csv"
  with open(output_path, 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)   
    filewriter.writerow(["Slice", "Number of Blob", "Average Size", "Circ", "AR", "Greenness", "Class"])

  for files in glob.glob(filepath):
  	with open(output_path, 'a') as outputFile:
  	  wr = csv.writer(outputFile, quoting=csv.QUOTE_ALL)
  	  attr = extractAttributes(files)
  	  wr.writerows([attr])
  print("done")

#batchProcessing("/Users/violet/Documents/18WI/CSC499/images/x16x16x4/*.png")
