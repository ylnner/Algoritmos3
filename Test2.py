import numpy as np
import math
import scipy.spatial
import time
import copy

def distanceDTW(signal_1, signal_2):	
	globalDTW = np.zeros((len(signal_1) + 1, len(signal_2) + 1))
	globalDTW[1:, 0] = math.inf
	globalDTW[0, 1:] = math.inf

	globalDTW[0, 0] = 0
	# print(globalDTW)
	for i in range(1, len(signal_1)+1):
		for j in range(1, len(signal_2)+1):							
			# aux = math.fabs(signal_1[i-1] - signal_2[j-1])
			# minimo = min(globalDTW[i-1, j], globalDTW[i, j-1], globalDTW[i-1, j-1])		
			globalDTW[i, j] = math.fabs(signal_1[i-1] - signal_2[j-1]) + min(globalDTW[i-1, j], globalDTW[i, j-1], globalDTW[i-1, j-1])
	return globalDTW[len(signal_1), len(signal_2)]


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

nameFile = 'treino.txt'
labelTrain, timeSeriesTrain = loadFile(nameFile)
nameFile = 'teste.txt'
labelTest, timeSeriesTest = loadFile(nameFile)


# from dtw import dtw
# euclidean_norm = lambda x, y: np.abs(x - y)
# a = [0.33333 , 0.29167 , 0.29167 , 0.27778 , 0.23611 , 0.22222 , 0.16667 , 0.13889 , 0.097222 , 0.083333 , 0.055556 , 0.069444 , 0.069444 , 0.055556 , 0.069444 , 0.083333 , 0.097222 , 0.097222 , 0.125 , 0.16667 , 0.15278 , 0.15278 , 0.20833 , 0.27778 , 0.33333 , 0.375 , 0.40278 , 0.44444 , 0.47222 , 0.55556 , 0.55556 , 0.56944 , 0.55556 , 0.54167 , 0.52778 , 0.55556 , 0.56944 , 0.59722 , 0.61111 , 0.625 , 0.61111 , 0.56944]
start = time.time()
for k in range(len(timeSeriesTrain)):		
	# dist = distanceDTW(a, timeSeriesTrain[j])
	#d = distanceDTW(timeSeriesTest[369], timeSeriesTrain[j])
	signal_1 = timeSeriesTest[369]
	signal_2 = timeSeriesTrain[k]


	# globalDTW = [[0 for x in range(len(signal_2) + 1)] for x in range(len(signal_1) + 1)]
	# for i in range(len(signal_1)):
	# 	globalDTW[i+1][0] = math.inf

	# for i in range(len(signal_2)):
	# 	globalDTW[0][i+1] = math.inf

	# globalDTW = np.array(globalDTW)

	globalDTW = np.zeros((len(signal_1) + 1, len(signal_2) + 1))
	
	globalDTW[1:, 0] = math.inf
	globalDTW[0, 1:] = math.inf

	# print((globalDTW.shape))

	
	globalDTW[0, 0] = 0
	# # print(globalDTW)
	for i in range(1, len(signal_1)+1):
		for j in range(1, len(signal_2)+1):							
			# aux = math.fabs(signal_1[i-1] - signal_2[j-1])
			# minimo = min(globalDTW[i-1, j], globalDTW[i, j-1], globalDTW[i-1, j-1])

			globalDTW[i, j] = math.fabs(signal_1[i-1] - signal_2[j-1]) + min(globalDTW[i-1, j], globalDTW[i, j-1], globalDTW[i-1, j-1])
	dist = globalDTW[len(signal_1), len(signal_2)]




end = time.time()
duration = end - start
print('duration: ', duration)


# durationGlobal = 0
# predictTotal = {}
# idx = 1

# for i in range(len(timeSeriesTest)):
# 	start = time.time()
# 	min_distance = math.inf
# 	predict = 0

# 	for j in range(len(timeSeriesTrain)):		
# 		dist = distanceDTW(timeSeriesTest[i], timeSeriesTrain[j])
# 		if dist < min_distance:
# 			min_distance = dist
# 			predict = labelTrain[j]

# 	predictTotal[i] = predict

# 	if predict == labelTest[i]:
# 		accuracy =+1
# 	else:
# 		accuracy =-1
# 	end = time.time()
# 	duration = end - start
# 	print('idx: ', idx)
# 	print('duration: ', duration)
# 	durationGlobal = durationGlobal	 + duration
# 	idx = idx +1
	
# accuracy = accuracy / len(timeSeriesTest)
# print('durationGlobal: ', durationGlobal)
# print('accuracy: ', accuracy)


# predictTotal = {}
# accuracy = 0
# for i in range(len(timeSeriesTest)):
# 	# idx2 = 0
# 	min_distance = math.inf
# 	predict = 0
# 	start = time.time()
# 	for j in range(len(timeSeriesTrain)):
		
# 		dist = distanceDTW(timeSeriesTest[i], timeSeriesTrain[j])
		
# 		if dist < min_distance:
# 			min_distance = dist
# 			predict = labelTrain[j]
			
# 	predictTotal[i] = predict

# 	if predict == labelTest[i]:
# 		accuracy =+1
# 	else:
# 		accuracy =-1
# 	end = time.time()
# 	duration = end - start
# 	print('duration element Testing: ', duration)

# accuracy = accuracy / len(timeSeriesTest)
	# ++idx
	

# x = [0, 4, 4, 0, -4, -4, 0]
# y = [1, 3, 4, 3, 1, -1, -2, -1, 0]

# x = np.array([1, 2, 3, 5, 5, 5, 6])
# y = np.array([1, 1, 2, 2, 3, 5])

# distanceDTW(timeSeries[0], timeSeries[1])



