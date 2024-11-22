import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

class Preprocess(object):
	def __init__(self):
		pass

	def clean_text(self, text):
		'''
		Args:
			text: string 
		Return:
        	cleaned_text: string
        '''
		# Remove HTML formatting
		soup = BeautifulSoup(text)
		cleaned_text = soup.get_text()

		# Remove non-alphabet characters such as punctuation or numbers and replace with ' '
		cleaned_text = re.sub('^\s+|\W+|[0-9]|\s+$',' ',cleaned_text).strip()

		# Remove leading and trailing whitespace
		cleaned_text = re.sub('^\s*|\s*$', "", cleaned_text)

		# Convert to lowercase
		cleaned_text = cleaned_text.lower()

		# Tokenize and remove english stop words
		stop_words = set(stopwords.words('english'))
		cleaned_text = [word for word in word_tokenize(cleaned_text) if word not in stop_words]

		# Rejoin remaining text into one string using " " as the word separator
		return ' '.join(cleaned_text)

	def clean_dataset(self, data):
		'''	
		Args:
			data: list of N strings
		Return:
			cleaned_data: list of cleaned N strings
		'''
		cleaned_data = [self.clean_text(string) for string in data]

		return cleaned_data

pp = Preprocess()

def clean_wos(x_train, x_test):
	'''
	ToDo: Clean both the x_train and x_test dataset using clean_dataset from Preprocess
	Input:
		x_train: list of N strings
		x_test: list of M strings
	Output:
		cleaned_text_wos_train: list of cleaned N strings
		cleaned_text_wos_test: list of cleaned M strings
	'''
	cleaned_text_wos_train = pp.clean_dataset(x_train)
	cleaned_text_wos_test = pp.clean_dataset(x_test)

	return cleaned_text_wos_train, cleaned_text_wos_test
