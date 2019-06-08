import numpy as np
import math
import numpy, scipy, scipy.spatial, matplotlib.pyplot as plt
import time



global globalDTW


def euclidianDistance(x, y):
	# return (x - y)**2
	d = scipy.spatial.distance.euclidean(x, y)
	# d = abs(x-y)	
	return d


def distanceDTW(signal_1, signal_2):
	global globalDTW
	globalDTW = np.zeros((len(signal_1) + 1, len(signal_2) + 1))
	globalDTW[1:, 0] = math.inf
	globalDTW[0, 1:] = math.inf

	
	globalDTW[0, 0] = 0
	# print(globalDTW)
	for i in range(1, len(signal_1) + 1):
		for j in range(1, len(signal_2) + 1):			
			aux = euclidianDistance(signal_1[i-1], signal_2[j-1])			
			minimo = min(globalDTW[i-1, j], globalDTW[i, j-1], globalDTW[i-1, j-1])		
			globalDTW[i, j] = aux + minimo

	return globalDTW[len(signal_1), len(signal_2)]


def loadFile(nameFile):
	data  = open(nameFile, 'r')

	label = []
	timeSeries = {}

	#numpy.array([[0,1,2,3], [2,3,4]], dtype=object)
	idx = 0
	for line in data.readlines():
		np_temp         = np.fromstring(line, dtype = float, sep = ' ')		
		label           = np.append(label, np_temp[0])
		np_temp         = np.delete(np_temp, 0)
		timeSeries[idx] = np_temp
		idx             = idx +1
	data.close()

	return label, timeSeries


nameFile = 'treino.txt'
labelTrain, timeSeriesTrain = loadFile(nameFile)

nameFile = 'teste.txt'
labelTest, timeSeriesTest = loadFile(nameFile)
# idx = 0

predictTotal = {}
accuracy = 0
for i in range(len(timeSeriesTest)):
	# idx2 = 0
	min_distance = math.inf
	predict = 0
	start = time.time()
	for j in range(len(timeSeriesTrain)):
		
		dist = distanceDTW(timeSeriesTest[i], timeSeriesTrain[j])
		
		if dist < min_distance:
			min_distance = dist
			predict = labelTrain[j]
			
	predictTotal[i] = predict

	if predict == labelTest[i]:
		accuracy =+1
	else:
		accuracy =-1
	end = time.time()
	duration = end - start
	print('duration element Testing: ', duration)

accuracy = accuracy / len(timeSeriesTest)
	# ++idx
	

# x = [0, 4, 4, 0, -4, -4, 0]
# y = [1, 3, 4, 3, 1, -1, -2, -1, 0]

# x = np.array([1, 2, 3, 5, 5, 5, 6])
# y = np.array([1, 1, 2, 2, 3, 5])

# distanceDTW(timeSeries[0], timeSeries[1])



