"""SENTIMENT ANALYSIS TRAINING MODULE"""

import time # Just to compare fit times
import numpy as np
import pickle
import sys
import json
import re
from preprocess import preproc

from pyspark import SparkContext
from pyspark.streaming import StreamingContext, DStream
from pyspark.sql import SQLContext, Row, SparkSession

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import GridSearchCV

from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer, CountVectorizer, HashingTF
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit 
from pyspark.mllib.evaluation import BinaryClassificationMetrics

# config
sc = SparkContext("local[2]", "NetworkWordCount")
spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)
sqc = SQLContext(sc)

global vectorizer
vectorizer = HashingTF(inputCol='Tweet', outputCol='features')
vectorizer.setNumFeatures(1000)

global sknb
global skbnb
global sksgd
global km
global sknb_model
global skbnb_model
global sksgd_model
global sknb_tuning
skbnb = BernoulliNB()
sksgd = SGDClassifier()
km = MiniBatchKMeans(n_clusters=2, batch_size=7600, init="k-means++")

def get_pred(tweet):
	#print('hi')
	if not tweet.isEmpty():
		df = spark.createDataFrame(tweet)
		label_list = df.select('label').collect()
		Y = [row.label for row in label_list]
		result = vectorizer.transform(df)
		feature_list = result.select('features').collect()
		X = [row.features.toArray() for row in feature_list]
		
		sknb_tuning = GridSearchCV(MultinomialNB(), {'alpha': [0.01, 1, 0.1, 0.001, 0.00001], 'fit_prior': [True, False]})
		sknb_tune = sknb_tuning.fit(X, Y)
		sknb = MultinomialNB(alpha=sknb_tune.best_params_['alpha'])
		sknb_model = sknb.partial_fit(X, Y, classes=np.unique(Y))
		pickle.dump(sknb, open('./models/NaiveBayes.sav', 'wb'))
		print('MultinomialNB: ', sknb_model.score(X, Y))
		
		skbnb_tuning = GridSearchCV(BernoulliNB(), {'alpha':[0.01, 1, 0.1, 0.001, 0.00001], 'binarize':[0.0, 1.0, 0.001, 0.0001, 0.00001]})
		skbnb_tune = skbnb_tuning.fit(X, Y)
		skbnb = BernoulliNB(alpha=skbnb_tune.best_params_['alpha'], binarize=skbnb_tune.best_params_['binarize'])
		skbnb_model = skbnb.partial_fit(X, Y, classes=np.unique(Y))
		pickle.dump(skbnb, open('./models/BernoulliBayes.sav', 'wb'))
		print('BernoulliNB: ', skbnb_model.score(X, Y))
		
		sksgd_tuning = GridSearchCV(SGDClassifier(), {'alpha':[1, 0.1, 0.01], 'epsilon':[1, 0.1, 0.01]})
		sksgd_tune = sksgd_tuning.fit(X, Y)
		sksgd = SGDClassifier(alpha=sksgd_tune.best_params_['alpha'], epsilon=sksgd_tune.best_params_['epsilon'])
		sksgd_model = skbnb.partial_fit(X, Y, classes=np.unique(Y))
		pickle.dump(sksgd, open('./models/SKSGD.sav', 'wb'))
		print('SKSGD: ', sksgd_model.score(X, Y))
		print()

lines = ssc.socketTextStream('localhost', 6100)
lines = lines.flatMap(lambda line: json.loads(line))
lines = lines.filter(lambda line: line[0] != 'S')
tweets = lines.map(lambda tweet: Row(label=float(tweet[0]),Tweet=preproc(tweet[2:]).split(' ')))
tweets.foreachRDD(get_pred)
ssc.start()
ssc.awaitTermination()

