'''
Om Arham Mukha Kamal Vaasinee Paapaatma Kshayam Kaari Vad Vad Vaagwaadinee Saraswati Aing Hreeng Namah Swaaha 
'''
import sys
import json
import re
from pyspark import SparkContext
from pyspark.streaming import StreamingContext, DStream
from pyspark.sql import SQLContext, Row, SparkSession
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# config
sc = SparkContext("local[2]", "NetworkWordCount")
ssc = StreamingContext(sc, 1)
sqc = SQLContext(sc)

def clean(x):
	x = x.replace('\\n', '')
	x = x.replace('\\', '')
	return x

def preproc(item):
	if len(item) > 2:
		for i in item[2:]:
			item[1] += i
		item = item[:2]
	# TODO preprocessing
	#removing punctuation, @, RT, making it lower case
	item[1] = re.sub('http\S+', '', item[1])
	item[1] = re.sub('@\w+', '', item[1])
	item[1] = re.sub('#', '', item[1])
	item[1] = re.sub('RT', '', item[1])
	item[1] = re.sub(':', '', item[1])
	item[1] = re.sub('",', '', item[1])
	item[1] = re.sub('\\n', '', item[1])
	'''item[1] = re.sub(')', '', item[1])
	item[1] = re.sub('(', '', item[1])
	item[1] = re.sub('[', '', item[1])
	item[1] = re.sub(']', '', item[1])
	item[1] = re.sub('{', '', item[1])
	item[1] = re.sub('}', '', item[1])'''
	item[1] = re.sub(r'[^\w\s]', '', item[1])
	item[1] = item[1].lower()
	
	# TODO lemmatization, stop word removal
	return item

lines = ssc.socketTextStream('localhost', 6100)
lines = lines.flatMap(lambda line: json.loads(line)).map(lambda x: x.split(','))
#lines.pprint()
preprocessed_lines = lines.map(lambda line: preproc(line))
#preprocessed_lines.pprint()
preprocessed_lines.foreachRDD(

ssc.start()
ssc.awaitTermination()
