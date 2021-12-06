'''Cluster Testing module'''
# Utility Modules
import pickle
import sys
import json
import numpy as np
from preprocess import preproc

# Spark boilerplate
from pyspark import SparkContext
from pyspark.streaming import StreamingContext, DStream
from pyspark.sql import SQLContext, Row, SparkSession

# Classifiers
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Vectorizer
from pyspark.ml.feature import HashingTF

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
global km_model
km_model = pickle.load(open('./models/KM.sav', 'rb'))

def get_pred(tweet):
	"""
		Prediction Driver
	"""
	if not tweet.isEmpty():
		df = spark.createDataFrame(tweet)
		label_list = df.select('label').collect()
		Y = [row.label for row in label_list]
		result = vectorizer.transform(df)
		feature_list = result.select('features').collect()
		X = [row.features.toArray() for row in feature_list]
		
		pred_labels = km_model.predict(X)
		print('accuracy score:', accuracy_score(Y, pred_labels))
		print()
		
# Driver Code
lines = ssc.socketTextStream('localhost', 6100)
lines = lines.flatMap(lambda line: json.loads(line))
lines = lines.filter(lambda line: line[0] != 'S')
tweets = lines.map(lambda tweet: Row(label=float(tweet[0]),Tweet=preproc(tweet[2:]).split(' ')))
tweets.foreachRDD(get_pred)
ssc.start()
ssc.awaitTermination()
