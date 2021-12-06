# SENTIMENT ANALYSIS

import time # Just to compare fit times
import numpy as np
import pickle
import sys
import json
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from pyspark import SparkContext
from pyspark.streaming import StreamingContext, DStream
from pyspark.sql import SQLContext, Row, SparkSession
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer, CountVectorizer, HashingTF

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import HashingVectorizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# hyperparams tuning
# from sklearn.model_selection import TuneSearchCV # :( says it's illegal
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit 
# Train Valid split better than crossValid in most cases
from pyspark.mllib.evaluation import BinaryClassificationMetrics
 
lemmatizer = WordNetLemmatizer()
stwords = stopwords.words('english')

morestwords = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
stwords += morestwords

# config
sc = SparkContext("local[2]", "NetworkWordCount")
spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)
sqc = SQLContext(sc)

global vectorizer
vectorizer = HashingTF(inputCol='Tweet', outputCol='features')
vectorizer.setNumFeatures(500)

global sknb
global skbnb
global sksgd
global sknb_model
global skbnb_model
global sksgd_model
sknb = MultinomialNB()
skbnb = BernoulliNB()
sksgd = SGDClassifier()
km = MiniBatchKMeans(n_clusters=2)

def preproc(item):
	#removing punctuation, @, RT, making it lower case
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

def get_pred(tweet):
	#print('hi')
	if not tweet.isEmpty():
		df = spark.createDataFrame(tweet)
		label_list = df.select('label').collect()
		Y = [row.label for row in label_list]
		result = vectorizer.transform(df)
		feature_list = result.select('features').collect()
		X = [row.features.toArray() for row in feature_list]
		sknb_model = sknb.partial_fit(X, Y, classes=np.unique(Y))
		pickle.dump(sknb, open('NaiveBayes.sav', 'wb'))
		print('MultinomialNB: ', sknb_model.score(X, Y))
		skbnb_model = skbnb.partial_fit(X, Y, classes=np.unique(Y))
		pickle.dump(skbnb, open('BernoulliBayes.sav', 'wb'))
		print('SKBNB: ', skbnb_model.score(X, Y))
		sksgd_model = sksgd.partial_fit(X, Y, classes=np.unique(Y))
		print('SKSGD: ', sksgd_model.score(X, Y))
		pickle.dump(sksgd, open('SKSGD.sav','wb'))
		print()

lines = ssc.socketTextStream('localhost', 6100)
lines = lines.flatMap(lambda line: json.loads(line))
lines = lines.filter(lambda line: line[0] != 'S')
tweets = lines.map(lambda tweet: Row(label=float(tweet[0]),Tweet=preproc(tweet[2:]).split(' ')))
tweets.foreachRDD(get_pred)
ssc.start()
ssc.awaitTermination()

