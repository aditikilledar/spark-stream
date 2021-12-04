'''
Om Arham Mukha Kamal Vaasinee Paapaatma Kshayam Kaari Vad Vad Vaagwaadinee Saraswati Aing Hreeng Namah Swaaha 
'''
import sys
import json
import re
from pyspark import SparkContext
from pyspark.streaming import StreamingContext, DStream
from pyspark.sql import SQLContext, Row, SparkSession
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import HashingVectorizer
 
lemmatizer = WordNetLemmatizer()
stwords = stopwords.words('english')

morestwords = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
stwords += morestwords

# config
sc = SparkContext("local[2]", "NetworkWordCount")
spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)
sqc = SQLContext(sc)

global model
model = NaiveBayes()
global vectorizer
vectorizer = HashingVectorizer(norm = None, alternate_sign = False)

def clean(x):
	x = x.replace('\\n', '')
	x = x.replace('\\', '')
	return x

def preprocess(record, spark):
	if not record.isEmpty():
		df = spark.createDataFrame(record) 
		df.show()

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
		nitem += ' ' + word
	#item[0] = re.sub('\'', '', item[0])
	
	return nitem

lines = ssc.socketTextStream('localhost', 6100)
lines = lines.flatMap(lambda line: json.loads(line))
text_dstream = lines.map(lambda tweet: tweet[2:])
#lines.pprint()
sentiment_dstream = lines.map(lambda tweet: tweet[0])
preprocessed_lines = text_dstream.map(lambda line: preproc(line))
#preprocessed_lines.foreachRDD(lambda rdd: rdd.collect())

vectorized_lines = preprocessed_lines.transform(lambda rdd: preprocess(rdd, spark))
vectorized_lines.pprint()


# TODO remove 'sentiment', 'tweet' (not manually)
#labelled_points = preprocessed_lines.map(lambda line: LabeledPoint(line[0], line[1]))
#labelled_points.pprint()
ssc.start()
ssc.awaitTermination()
