'''
Om Arham Mukha Kamal Vaasinee Paapaatma Kshayam Kaari Vad Vad Vaagwaadinee Saraswati Aing Hreeng Namah Swaaha 
'''
import sys
import json
import re
from pyspark import SparkContext
from pyspark.streaming import StreamingContext, DStream
from pyspark.sql import SQLContext, Row, SparkSession

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
		item.pop()
	# TODO preprocessing
	return item

lines = ssc.socketTextStream('localhost', 6100)
lines = lines.flatMap(lambda line: json.loads(line)).map(lambda x: x.split(','))
lines.pprint()
lines.map(lambda line: preproc(line)).pprint()

ssc.start()
ssc.awaitTermination()
