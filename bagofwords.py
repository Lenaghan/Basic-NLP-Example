import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import vstack, csr_matrix

class OHE_BOW(object): 
	def __init__(self):
		# initialize instances of encoders/vectorizers for use in fit and transform
		self.vocab_size = None
		self.oh = OneHotEncoder()
		self.cv = CountVectorizer()
		self.tv = TfidfVectorizer()

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
		words = np.array(words).reshape(-1, 1)
		onehotencoded = self.oh.transform(words).toarray()
		return onehotencoded

	# process data in batches using OneHotEncoder
	def bow_transform(self, data, batch_size=10):
		'''
		Transform the given data into a bag of words representation with OneHotEncoder.
		Args:
			data: list of N strings
			batch_size: int, size of each batch for processing
		Return:
			bow: (N, D) sparse matrix
		'''	
		batch_bow = []
		for i in range(0, len(data), batch_size):
			batch = data[i:i + batch_size]
			string_list = self.split_text(batch) # separate the strings into lists of words

			for word_list in string_list:
				try: # check for empty strings
					one_bow = self.onehot(word_list).sum(axis=0) # transform into bag of words
				except: one_bow = np.zeros(self.vocab_size) # array of zeros if empty string
				batch_bow.append(csr_matrix(one_bow))
		
		bow = vstack(batch_bow)
		return bow
	
	# process data in batches using CountVectorizer
	def cv_bow_transform(self, data, fit=False, batch_size=10):
		'''
		Transform the given data into a bag of words representation using CountVectorizer.
		Args:
			data: list of N strings
			fit: boolean, whether to fit the CountVectorizer instance to the data
			batch_size: int, size of each batch for processing
		Return:
			bow: (N, D) sparse matrix
		'''		
		# If fitting is required, fit on all data first to establish vocabulary
		if fit:
			self.cv.fit(data)
		
		batch_bow = []
		for i in range(0, len(data), batch_size):
			batch = data[i:i + batch_size]
			cv_bow = self.cv.transform(batch)  # transform into bag of words
			batch_bow.append(cv_bow)
		
		bow = vstack(batch_bow)
		return bow
	
	def tfidf_weights_transform(self, data, fit=False, batch_size=10):
		'''
		Transform the given data into a TF-IDF representation using TfidfVectorizer.
		Args:
			data: list of N strings
			fit: boolean, whether to fit the TfidfVectorizer instance to the data
			batch_size: int, size of each batch for processing
		Return:
			bow: (N, D) sparse matrix with TF-IDF weights
		'''        
		# If fitting is required, fit on all data first to establish vocabulary
		if fit:
			self.tv.fit(data)
		
		batch_bow = []
		for i in range(0, len(data), batch_size):
			batch = data[i:i + batch_size]
			tfidf_bow = self.tv.transform(batch)  # transform into TF-IDF weighted features
			batch_bow.append(tfidf_bow)
		
		bow = vstack(batch_bow)
		return bow