'''Module for preprocessing'''
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stwords = stopwords.words('english')
morestwords = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
stwords += morestwords

def preproc(item):
	'''
		removes stop words and punctuation from string and lemmatizes the words
		returns: list of words
	'''
	item = re.sub('http\S+', '', item)
	item = re.sub('@\w+', '', item)
	item = re.sub('#', '', item)
	item = re.sub('RT', '', item)
	item = re.sub(':', '', item)
	item = re.sub('",', '', item)
	item = re.sub('\\n', '', item)
	item = re.sub(r'[^\w\s]', ' ', item)
	item = item.lower()
	item = re.sub(r'\d+', '', item)
	item = [word for word in item.split(' ') if word not in stwords]
	item = [lemmatizer.lemmatize(word) for word in item if word != '']
	nitem = ''
	for word in item:
		nitem += word + ' '
	
	return nitem
