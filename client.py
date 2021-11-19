'''
Om Arham Mukha Kamal Vaasinee Paapaatma Kshayam Kaari Vad Vad Vaagwaadinee Saraswati Aing Hreeng Namah Swaaha 
'''
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
import sys

# config
sc = SparkContext("local[2]", "NetworkWordCount")
ssc = StreamingContext(sc, 1)

mt = sc.emptyRDD()

def clean(x):
	x = x.replace('\\n', '')
	x = x.replace('\\', '')
	print(x)
	return x

lines = ssc.socketTextStream("localhost", 6100)
a = lines.map(lambda x: clean(x)).reduce(lambda x, y : x.join(y))
#records = lines.flatMap(lambda line: line.split('\\n"')).map(lambda x: x.split(','))
a.pprint(50)
ssc.start()
ssc.awaitTermination()
print(a)

