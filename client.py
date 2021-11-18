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

lines = ssc.socketTextStream("localhost", 6100)
#print(lines)
# Count each word in each batch
records = lines.flatMap(lambda line: line.split('\\n",')).map(lambda x: x.split(','))
#records = lines.foreachRDD(lambda line: line)
records.pprint(20)
#for record in records:
#	print(record)

# Print the first ten elements of each RDD generated in this DStream to the console
#counts.pprint()

ssc.start()
ssc.awaitTermination()
