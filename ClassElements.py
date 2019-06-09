import numpy as np

class ClassElements:
	def __init__(self, classe):
		self.classe   = classe
		self.elements = []

	def setClasse(self, classe):
		self.classe = classe

	def setElements(self, elements):
		self.elements = elements

	def getClasse(self):
		return self.classe

	def getElements(self):
		return self.elements

	def addElement(self, element):
		# self.elements = np.append(self.elements, element)
		self.elements.append(element)
		# classElementsAll.append

	def printElements(self):
		a = ' '
		for i in range(len(self.elements)):
			# print()
			a += str(self.elements[i]) + ' '
			# print(self.elements[i])
		print(a)