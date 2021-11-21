'''
Om Arham Mukha Kamal Vaasinee Paapaatma Kshayam Kaari Vad Vad Vaagwaadinee Saraswati Aing Hreeng Namah Swaaha 
'''
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext, Row, SparkSession
import sys

# config
sc = SparkContext("local[2]", "NetworkWordCount")
ssc = StreamingContext(sc, 1)

def clean(x):
	x = x.replace('\\n', '')
	x = x.replace('\\', '')
	return x

lines = ssc.socketTextStream("localhost", 6100)
a = lines.map(lambda x: clean(x)).reduce(lambda x, y : x.join(y))

def getSparkSessionInstance(sparkConf):
    if ('sparkSessionSingletonInstance' not in globals()):
        globals()['sparkSessionSingletonInstance'] = SparkSession\
            .builder\
            .config(conf=sparkConf)\
            .getOrCreate()
    return globals()['sparkSessionSingletonInstance']

def process(time, rdd):
    print("========= %s =========" % str(time))

    # Get the singleton instance of SparkSession
    spark = getSparkSessionInstance(rdd.context.getConf())

    # Convert RDD[String] to RDD[Row] to DataFrame
    rowRdd = rdd.map(lambda w: Row(word=w))
    wordsDataFrame = spark.createDataFrame(rowRdd)

    # Creates a temporary view using the DataFrame.
    wordsDataFrame.createOrReplaceTempView("words")

    # Do word count on table using SQL and print it
    wordCountsDataFrame = \
        spark.sql("select word, count(*) as total from words group by word")
    wordCountsDataFrame.show()
            
lines = ssc.socketTextStream('localhost', 6100)
words = lines.map(lambda line: line)
words.pprint()
words.foreachRDD(process)
ssc.start()
ssc.awaitTermination()
'''mt = sc.emptyRDD()

def clean(x):
	x = x.replace('\\n', '')
	x = x.replace('\\', '')
	return x

lines = ssc.socketTextStream("localhost", 6100)
a = lines.map(lambda x: clean(x)).reduce(lambda x, y : x.join(y))

a.pprint(50) '''



