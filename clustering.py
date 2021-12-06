"""CLUSTERING ANALYSIS TRAINING MODULE"""
# Utility Modules
import numpy as np
import pickle
import sys
import json
from preprocess import preproc

# Spark boilerplate
from pyspark import SparkContext
from pyspark.streaming import StreamingContext, DStream
from pyspark.sql import SQLContext, Row, SparkSession

# Model
from sklearn.cluster import MiniBatchKMeans

# Vectorizer
from pyspark.ml.feature import HashingTF

# Hyperparameter Tuning
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score

# config
sc = SparkContext("local[2]", "NetworkWordCount")
spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)
sqc = SQLContext(sc)

global vectorizer
vectorizer = HashingTF(inputCol='Tweet', outputCol='features')
vectorizer.setNumFeatures(1000)

global km
km = MiniBatchKMeans(n_clusters=2, init="k-means++")
transformer = IncrementalPCA(n_components = 2, batch_size=7600)

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
		transformer.partial_fit(X)
		transformed_X = transformer.fit_transform(X)
		km_model = km.partial_fit(X)
		pickle.dump(km, open('./models/KM.sav', 'wb'))
		print('accuracy score:', accuracy_score(Y, km_model.labels_))

lines = ssc.socketTextStream('localhost', 6100)
lines = lines.flatMap(lambda line: json.loads(line))
lines = lines.filter(lambda line: line[0] != 'S')
tweets = lines.map(lambda tweet: Row(label=float(tweet[0]),Tweet=preproc(tweet[2:]).split(' ')))
tweets.foreachRDD(get_pred)
ssc.start()
ssc.awaitTermination()
