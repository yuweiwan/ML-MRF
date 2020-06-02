import numpy as np 
import sys








# Input files
gold_standard_file = sys.argv[1]
segmentation_file = sys.argv[2]



# Load in data
gold_standard = np.loadtxt(gold_standard_file)
segmentation = np.loadtxt(segmentation_file)

total_number_of_pixels = gold_standard.shape[0]*gold_standard.shape[1]
correctly_labeled_pixels = np.sum(gold_standard == segmentation)

accuracy = correctly_labeled_pixels/total_number_of_pixels

print('Accuracy: ' + str(accuracy))