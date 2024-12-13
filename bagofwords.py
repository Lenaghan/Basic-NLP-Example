import numpy as np
from sklearn.preprocessing import OneHotEncoder

class OHE_BOW(object): 
	def __init__(self):
		'''
		Initializes instance of OneHotEncoder in self.oh for use in fit and transform
		'''
		self.vocab_size = None
		self.oh = OneHotEncoder()

	def split_text(self, data):
		'''
		Separates each string into a list of individual words
		Args:
			data: list of N strings
		
		Return:
			data_split: list of N lists of individual words from each string
		'''
		data_split = [i.split() for i in data]
		return data_split
	
	def flatten_list(self, data):
		'''
		Flattens a list of list of words into a single list
		Args:
			data: list of N lists of W_i words 
		
		Return:
			data_split: (W,) numpy array of words, where W is the sum of the number of W_i words
			in each of the list of words		
		'''
		data_split = [i for items in data for i in items]
		return data_split

	def fit(self, data):
		'''
		Fits the initialized instance of OneHotEncoder to the given data
		Args:
			data: list of N strings 
		'''
		data = self.split_text(data)
		data = self.flatten_list(data)
		data = np.array(list(set(data)))
		self.vocab_size = len(data) # set to the number of unique words in the given data corpus
		self.oh.fit_transform(data.reshape(self.vocab_size, -1))

	def onehot(self, words):
		'''
		Encodes a list of words into one hot encoding format
		Args:
			words: list of W_i words from a string
		
		Return:
			onehotencoded: (W_i, D) numpy array where:
				W_i is the number of words in the current input list i
				D is the vocab size
		'''
		text = np.array(words)
		onehotencoded = self.oh.transform(text.reshape(-1, 1)).toarray()
		return onehotencoded

	def oneHotEncoding(self, row):
		try:
			return self.onehot(row).sum(axis=0)
		except:
			return np.zeros(self.vocab_size)    

	def transform(self, data):
		'''
		Uses the already fitted instance of OneHotEncoder to transform the given 
		data into a bag of words representation.
		Args:
			data: list of N strings
		
		Return:
			bow: (N, D) numpy array
		'''
		bow = []
		data = self.split_text(data)
		for i in data:
			try:
				one_bow = self.onehot(i).sum(axis=0)
			except: one_bow = np.zeros(self.vocab_size)
			bow.append(one_bow)

		bow = np.array(bow)
		return bow