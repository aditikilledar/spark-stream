'''Testing module'''
# Utility Modules
import pickle
import sys
import json
import numpy as np
import re
from preprocess import preproc

# Spark boilerplate
from pyspark import SparkContext
from pyspark.streaming import StreamingContext, DStream
from pyspark.sql import SQLContext, Row, SparkSession

# Classifiers
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import classification_report, confusion_matrix

# Vectorizer
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer, CountVectorizer, HashingTF
from sklearn.feature_extraction.text import HashingVectorizer

# Performance Metrics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# config
sc = SparkContext("local[2]", "NetworkWordCount")
spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)
sqc = SQLContext(sc)

# Vectorizer
global vectorizer
vectorizer = HashingTF(inputCol='Tweet', outputCol='features')
vectorizer.setNumFeatures(1000)

# Models
global sknb_model
global skbnb_model
global sksgd_model
global km_model
sknb_model = pickle.load(open('./models/NaiveBayes.sav','rb'))
skbnb_model = pickle.load(open('./models/BernoulliBayes.sav','rb'))
sksgd_model = pickle.load(open('./models/SKSGD.sav', 'rb'))
km_model = pickle.load(open('./models/KMeans.sav', 'rb'))

def get_pred(tweet):
	"""
		Testing Driver
	"""
	if not tweet.isEmpty():
		df = spark.createDataFrame(tweet)
		label_list = df.select('label').collect()
		Y = [row.label for row in label_list]
		result = vectorizer.transform(df)
		feature_list = result.select('features').collect()
		X = [row.features.toArray() for row in feature_list]

		print('----------Multinomial Naive Bayes----------')
		#print('NaiveBayes: ',sknb_model.score(X, Y))
		nb_report = classification_report(Y, sknb_model.predict(X), labels=np.unique(Y))
		#nb_conf = confusion_matrix(Y, sknb_model.predict(X), labels=np.unique(Y))
		#print('Confusion_Matrix: ', nb_conf)
		print('Report: ', nb_report)

		print('----------Bernoulli Naive Bayes----------')
		#print('BernoulliBayes: ',skbnb_model.score(X, Y))
		sknb_report = classification_report(Y, sknb_model.predict(X), labels=np.unique(Y))
		#skbnb_conf = confusion_matrix(Y, sknb_model.predict(X), labels=np.unique(Y))
		#print('Confusion_Matrix: ', skbnb_conf)
		print('Report: ', skbnb_report)
		
		print('----------SGD----------')
		#print('SKSGD: ', sksgd_model.score(X, Y))
		sksgd_report = classification_report(Y, sksgd_model.predict(X), labels=np.unique(Y))
		#sksgd_conf = confusion_matrix(Y, sksgd_model.predict(X), labels=np.unique(Y))
		#print('Confusion Matrix: ', sksgd_conf)
		print('Report: ', sksgd_report)
		print()
		
# Driver Code
lines = ssc.socketTextStream('localhost', 6100)
lines = lines.flatMap(lambda line: json.loads(line))
lines = lines.filter(lambda line: line[0] != 'S')
tweets = lines.map(lambda tweet: Row(label=float(tweet[0]),Tweet=preproc(tweet[2:]).split(' ')))
tweets.foreachRDD(get_pred)
ssc.start()
ssc.awaitTermination()

