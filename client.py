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
from pyspark.sql.functions import lit

from pyspark.ml.classification import NaiveBayes, DecisionTreeClassifier, LogisticRegression, GBTClassifier, LinearSVC, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
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
vectorizer.setNumFeatures(10)

global sknb
global skbnb
global sksgd
global sknb_model
global skbnb_model
global sksgd_model
sknb = MultinomialNB()
skbnb = BernoulliNB()
sksgd = SGDClassifier()

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
		print()
		
		'''
		parameters = {'alpha': [1e-4, 1e-1, 1], 'epsilon':[0.01, 0.1]}
		
		tune_search = TuneGridSearchCV(  # <-------- does not work for some reason, worth exploring
    		SGDClassifier(),
    		parameters,
    		early_stopping=True,
    		max_iters=10
		)

		start = time.time()
		tune_search.fit(X, Y)
		end = time.time()
		print("Tune Fit Time:", end - start)
		pred = tune_search.predict(X)
		accuracy = np.count_nonzero(np.array(pred) == np.array(Y)) / len(pred)
		print("Tune Accuracy:", accuracy)
		'''
		'''
		paramGrid = ParamGridBuilder()\
    		.addGrid(lr.regParam, [0.1, 0.01]) \
    		.addGrid(lr.fitIntercept, [False, True])\
    		.addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    		.build()

# In this case the estimator is simply the linear regression.
# A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
		tvs = TrainValidationSplit(estimator=,
                       estimatorParamMaps=paramGrid,
                       evaluator=RegressionEvaluator(),
                           # 80% of the data will be used for training, 20% for validation.
                       trainRatio=0.8)

# Run TrainValidationSplit, and choose the best set of parameters.
		# model = tvs.fit(train)
		tvs_model = tvs.partial_fit(X, Y, classes=np.unique(Y))
		print('TVS SKGSCD: ', tvs.score(X, Y))
		print()
		'''

lines = ssc.socketTextStream('localhost', 6100)
lines = lines.flatMap(lambda line: json.loads(line))
lines = lines.filter(lambda line: line[0] != 'S')
tweets = lines.map(lambda tweet: Row(label=float(tweet[0]),Tweet=preproc(tweet[2:]).split(' ')))
tweets.foreachRDD(get_pred)
ssc.start()
ssc.awaitTermination()
