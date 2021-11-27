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

def clean(x):
	x = x.replace('\\n', '')
	x = x.replace('\\', '')
	return x

def preprocess(record, spark):
	'''if not record.isEmpty():
		df = spark.createDataFrame(record) 
		df.show()'''
		#model.train(record)

def preproc(item):
	if len(item) > 2:
		for i in item[2:]:
			item[1] += i
		item = item[:2]
	#removing punctuation, @, RT, making it lower case
	item[1] = re.sub('http\S+', '', item[1])
	item[1] = re.sub('@\w+', '', item[1])
	item[1] = re.sub('#', '', item[1])
	item[1] = re.sub('RT', '', item[1])
	item[1] = re.sub(':', '', item[1])
	item[1] = re.sub('",', '', item[1])
	item[1] = re.sub('\\n', '', item[1])
	item[1] = re.sub(r'[^\w\s]', ' ', item[1])
	item[1] = item[1].lower()
	item[1] = re.sub(r'\d+', '', item[1])
	item[1] = [word for word in item[1].split(' ') if word not in stwords]
	item[1] = [lemmatizer.lemmatize(word) for word in item[1] if word != '']
	#item[0] = re.sub('\'', '', item[0])
	print(f'c{item[0]}c')
	if item[0] != 'Sentiment':
		item[0] = float(item[0])
	return item

lines = ssc.socketTextStream('localhost', 6100)
lines = lines.flatMap(lambda line: json.loads(line)).map(lambda x: x.split(','))
preprocessed_lines = lines.map(lambda line: preproc(line))
#preprocessed_lines.pprint()
#preprocessed_lines.foreachRDD(lambda rdd: preprocess(rdd, spark))
# TODO remove 'sentiment', 'tweet' (not manually)
labelled_points = preprocessed_lines.map(lambda line: LabeledPoint(line[0], line[1]))
labelled_points.pprint()
ssc.start()
ssc.awaitTermination()
