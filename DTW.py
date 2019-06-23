import numpy as np
import math
import os
import time
import sys

# def distanceDTW(signal_1, signal_2):	
# 	globalDTW = np.zeros((len(signal_1) + 1, len(signal_2) + 1))
# 	globalDTW[1:, 0] = math.inf
# 	globalDTW[0, 1:] = math.inf

# 	globalDTW[0, 0] = 0
# 	# print(globalDTW)
# 	for i in range(1, len(signal_1)+1):
# 		for j in range(1, len(signal_2)+1):							
# 			# aux = math.fabs(signal_1[i-1] - signal_2[j-1])
# 			# minimo = min(globalDTW[i-1, j], globalDTW[i, j-1], globalDTW[i-1, j-1])		
# 			globalDTW[i, j] = math.fabs(signal_1[i-1] - signal_2[j-1]) + min(globalDTW[i-1, j], globalDTW[i, j-1], globalDTW[i-1, j-1])
# 	return globalDTW[len(signal_1), len(signal_2)]


def loadFile(nameFile):
	global classElementsAll	
	data       = open(nameFile, 'r')	
	label      = []
	timeSeries = {}
	
	idx = 0
	for line in data.readlines():		
		np_temp         = np.fromstring(line, dtype = float, sep = ' ')
		label           = np.append(label, np_temp[0])
		auxLabel		= np_temp[0]
		np_temp         = np.delete(np_temp, 0)	
		timeSeries[idx] = np_temp
		idx             = idx +1		
	data.close()

	# print('label: ', label)
	# print('timeSeries: ', timeSeries)
	return label, timeSeries



def classification(labelTrain, timeSeriesTrain, labelTest, timeSeriesTest):	
	if os.path.exists("Classification.txt"):
		os.remove("Classification.txt")

	file_output = open("Classification.txt", "a+")


	durationGlobal = 0
	idx            = 1
	accuracy       = 0
	start          = time.time()
	for m in range(len(timeSeriesTest)):
		min_distance = math.inf
		predict = 0
		signal_1 = timeSeriesTest[m]

		for k in range(len(timeSeriesTrain)):					
			signal_2 = timeSeriesTrain[k]


			# This process DTW distance between two signals. 
			globalDTW        = np.zeros((len(signal_1) + 1, len(signal_2) + 1))			
			globalDTW[1:, 0] = math.inf
			globalDTW[0, 1:] = math.inf
			globalDTW[0, 0]  = 0

			for i in range(1, len(signal_1)+1):
				for j in range(1, len(signal_2)+1):										
					globalDTW[i, j] = math.fabs(signal_1[i-1] - signal_2[j-1]) + min(globalDTW[i-1, j], globalDTW[i, j-1], globalDTW[i-1, j-1])
			dist = globalDTW[len(signal_1), len(signal_2)]

			# Save the min distance
			if dist < min_distance:
				min_distance = dist
				predict = labelTrain[k]

		file_output.write("Number of series: " + str(idx) + '	Classification: '+ str(int(predict))+ '	True Class: '+ str(int(labelTest[m]))+ os.linesep)
		
		if predict == labelTest[m]:
			accuracy = accuracy + 1	
		print('Processing serie ', idx)	
		idx = idx +1

	end = time.time()
	duration = end - start	

	accuracy = accuracy / len(timeSeriesTest)
	print('accuracy: ', accuracy)
	print('Global Time(seg): ', round(duration,4))

	file_output.write("Accuracy: " + str(accuracy) + os.linesep)
	file_output.write("Global Time(seg): " + str(round(duration,4)) + os.linesep)
	file_output.close()


if __name__ == '__main__':	
	if len(sys.argv) == 3:
		print('Reading files...')
		nameFile = sys.argv[1]
		labelTrain, timeSeriesTrain = loadFile(nameFile)
		nameFile = sys.argv[2]
		labelTest, timeSeriesTest = loadFile(nameFile)

		classification(labelTrain, timeSeriesTrain, labelTest, timeSeriesTest)
	else:
		print('You need to specify the training and testing files.')