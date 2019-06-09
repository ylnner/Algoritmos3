import numpy as np
import math
import scipy.spatial
import time
import copy
from ClassElements import ClassElements


global globalDTW
global chains
global classElementsAll
classElementsAll = []


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
		added = False		
		if not classElementsAll:
			print('')
		else:
			for i in range(len(classElementsAll)):
				if classElementsAll[i].getClasse() == auxLabel:				
					classElementsAll[i].addElement(np_temp)
					added = True
					break

		if added == False:
			newClassElement = ClassElements(auxLabel)
			newClassElement.addElement(np_temp)
			classElementsAll.append(newClassElement)
	data.close()	
	return label, timeSeries

def getSubTrain(label):
	global classElementsAll
	subTrain = {}
	labelTrain = {}
	for i in range(len(classElementsAll)):
		if classElementsAll[i].getClasse() != label:			
			subTrain[classElementsAll[i].getClasse()] = classElementsAll[i].getElements()
		else:
			labelTrain[label] = classElementsAll[i].getElements()
	return subTrain, labelTrain


def makeChain(item, itemLabel)
	chain    = {}
	stop     = False
	minLabel = itemLabel
	subTrain,labelTrain = getSubTrain(minLabel)
	itemSearch = item
	for label, data in subTrain.items():
		minDist  = math.inf
		minLabel = 0
		minIdx   = -1
		for idx in range(len(data)):
			dist = distanceDTW(data[idx], itemSearch)		
			if dist < minDist:
				minDist  = dist
				minLabel = label
				minIdx   = idx

	if minLabel in chain:
		# visitados = chain[minLabel]			
		aux = chain[minLabel]
		aux.append(minIdx)
		del chain[minLabel]
		chain[minLabel] = aux
	else:
		aux = [minIdx]
		chain[minLabel] = aux


	oldLabel = -1
	oldIdx   = -1

	minLabel = itemLabel
	while not stop:
		subTrain,labelTrain = getSubTrain(minLabel)
		itemSearch = item
		for label, data in subTrain.items():
			minDist  = math.inf
			minLabel = 0
			minIdx   = -1
			for idx in range(len(data)):
				dist = distanceDTW(data[idx], itemSearch)		
				if dist < minDist:
					minDist  = dist
					minLabel = label
					minIdx   = idx

		print('minLabel: ', minLabel)
		if minLabel in chain:
			# visitados = chain[minLabel]
			aux = chain[minLabel]
			aux.append(minIdx)
			del chain[minLabel]
			chain[minLabel] = aux
		else:
			aux = [minIdx]
			chain[minLabel] = aux


		# visitados = chain[minLabel]
		# for i in range(len(visitados)):
		# 	if visitados[i] == minIdx:
		# 		stop = True
		# 		break
		# if stop == False:
		# 	aux = chain[minLabel]
		# 	aux.append(minIdx)
		# 	del chain[minLabel]
		# 	chain[minLabel] = aux	


def makeChain2(item, itemLabel, subTrain, labelTrain):
	oldClass = itemLabel
	chain    = {}
	stop     = False

	maxWeigth = 0	
	for label, data in subTrain.items():
		if len(data) > maxWeigth:
			maxWeigth = len(data)

	# print('labelTrain')
	# print(labelTrain)
	matrixDistance      = np.zeros((len(subTrain), maxWeigth))
	matrixDistanceLabel = np.zeros((1, len(labelTrain[itemLabel])))
	# print('matrixDistanceLabel: ')
	# print(matrixDistanceLabel)

	dictDistancesOrder = {}

	idxLabel = 0
	for label, data in subTrain.items():		
		# Analizo para cada clase que no sea la de item (itemLabel)
		for idx in range(len(data)):
			dist = matrixDistance[idxLabel, idx]
			if dist == 0:
				dist = distanceDTW(data[idx], item)
				matrixDistance[idxLabel, idx] = dist		
		
		arrayTemp = []
		for i in range(len(matrixDistance[idxLabel])):
			if matrixDistance[idxLabel][i] != 0:
				arrayTemp.append(matrixDistance[0][i])

		sortedIndex = sorted(range(len(arrayTemp)), key=lambda k: arrayTemp[k])
		dictDistancesOrder[label] = sortedIndex
		idxLabel += 1



	data = labelTrain[itemLabel]
	for idx in range(len(data)):
		dist = matrixDistanceLabel[0, idx]
		if dist == 0:
			dist = distanceDTW(data[idx], item)
			matrixDistanceLabel[0, idx] = dist

	arrayTemp = []
	for i in range(len(matrixDistanceLabel[0])):
		if matrixDistanceLabel[0][i] != 0:
			arrayTemp.append(matrixDistanceLabel[0][i])

	sortedIndex = sorted(range(len(arrayTemp)), key=lambda k: arrayTemp[k])
	dictDistancesOrder[itemLabel] = sortedIndex

	
		# Analizar otras clases

		# Analizar mi clase





	# arr2D = np.array(matrixDistance)
	# result = np.where(arr2D == np.amin(arr2D))
	# print('Tuple of arrays returned : ', result)

	# print('matrixDistance')
	# print(matrixDistance[0])
	# arrayTemp = []
	# for i in range(len(matrixDistance[0])):
	# 	if matrixDistance[0][i] != 0:
	# 		arrayTemp.append(matrixDistance[0][i])

	# sortedIndex = sorted(range(len(arrayTemp)), key=lambda k: arrayTemp[k])


	# arrayTemp = sorted(arrayTemp)
	# print(sorted(pyList))
	# print('arrayTemp')
	# print(arrayTemp)
	# print(sorted(arrayTemp))
	print('dictDistancesOrder')
	print(dictDistancesOrder)



		# if dist < minDist:
		# 	minDist  = dist
		# 	minLabel = label
		# 	minIdx   = idx
		# idxLabel += 1



	# calcula = 0	
	# while not stop:
	# 	idxLabel = 0
		
	# 	# Buscando en las otras clases
	# 	for label, data in subTrain.items():
	# 		minDist  = math.inf
	# 		minLabel = 0
	# 		minIdx   = -1
	# 		# Analizo para cada clase que no sea la de item (itemLabel)
	# 		for idx in range(len(data)):
	# 			dist = matrixDistance[idxLabel, idx]
	# 			if dist == 0:					
	# 				calcula +=1
	# 				dist = distanceDTW(data[idx], item)
	# 				matrixDistance[idxLabel, idx] = dist
				
	# 			# print('Fin Calculando')
	# 			if dist < minDist:
	# 				minDist  = dist
	# 				minLabel = label
	# 				minIdx   = idx
	# 		idxLabel += 1


	# 	if minLabel in chain:
	# 		visitados = chain[minLabel]
	# 		for i in range(len(visitados)):
	# 			if visitados[i] == minIdx:
	# 				stop = True
	# 				break
	# 		if stop == False:
	# 			aux = chain[minLabel]
	# 			aux.append(minIdx)
	# 			del chain[minLabel]
	# 			chain[minLabel] = aux				
	# 	else:
	# 		aux = [minIdx]
	# 		chain[minLabel] = aux

	# 	if stop == False:
	# 		# Buscando en mi clase
	# 		data = labelTrain[itemLabel]
	# 		for idx in range(len(data)):
	# 			dist = matrixDistanceLabel[0, idx]
	# 			if dist == 0:					
	# 				calcula +=1
	# 				dist = distanceDTW(data[idx], item)
	# 				matrixDistanceLabel[0, idx] = dist
				
	# 			# print('Fin Calculando')
	# 			if dist < minDist:
	# 				minDist  = dist					
	# 				minIdx   = idx

	# 		if itemLabel in chain:
	# 			visitados = chain[itemLabel]
	# 			for i in range(len(visitados)):
	# 				if visitados[i] == minIdx:
	# 					stop = True
	# 					break

	# 			if stop == False:
	# 				aux = chain[itemLabel]
	# 				aux.append(minIdx)
	# 				del chain[itemLabel]
	# 				chain[itemLabel] = aux
	# 		else:
	# 			aux = [minIdx]
	# 			chain[itemLabel] = aux
	# print(chain)
	# return chain

def TKNN(train, test, labelTrain, labelTest):
	for i in range(len(train)):
		item     = train[i]		
		subTrain, labelTrain = getSubTrain(labelTrain[i])
		start = time.time()
		chain    = makeChain(item, labelTrain[i], subTrain)
		end = time.time()
		duration = end - start
		print('duration: ', duration)



nameFile = 'treino.txt'
labelTrain, timeSeriesTrain = loadFile(nameFile)

# classElementsAll
# print(classElementsAll)

# for i in range(len(classElementsAll)):
# 	print('label: ', classElementsAll[i].getClasse())
# 	classElementsAll[i].printElements()
# TKNN(timeSeriesTrain, 0, labelTrain, 0)

a = [0.33333 , 0.29167 , 0.29167 , 0.27778 , 0.23611 , 0.22222 , 0.16667 , 0.13889 , 0.097222 , 0.083333 , 0.055556 , 0.069444 , 0.069444 , 0.055556 , 0.069444 , 0.083333 , 0.097222 , 0.097222 , 0.125 , 0.16667 , 0.15278 , 0.15278 , 0.20833 , 0.27778 , 0.33333 , 0.375 , 0.40278 , 0.44444 , 0.47222 , 0.55556 , 0.55556 , 0.56944 , 0.55556 , 0.54167 , 0.52778 , 0.55556 , 0.56944 , 0.59722 , 0.61111 , 0.625 , 0.61111 , 0.56944]
st, labelTrain = getSubTrain(1)
start = time.time()
makeChain(a, 1, st, labelTrain)
end = time.time()
duration = end - start
print('duration: ', duration)

# nameFile = 'teste.txt'
# labelTest, timeSeriesTest = loadFile(nameFile)
# # idx = 0

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



