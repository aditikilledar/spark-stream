from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.types import StructType,StructField, StringType, IntegerType
#from textblob import TextBlob

sc = SparkContext("local[2]", "NetworkWordCount")
ssc = StreamingContext(sc, 1)

def preprocessing(lines):
    words = lines.select(explode(split(lines.value, '\\n"')).alias("word"))
    '''words = words.na.replace('', None)
    words = words.na.drop()
    words = words.withColumn('word', F.regexp_replace('word', r'http\S+', ''))
    words = words.withColumn('word', F.regexp_replace('word', '@\w+', ''))
    words = words.withColumn('word', F.regexp_replace('word', '#', ''))
    words = words.withColumn('word', F.regexp_replace('word', 'RT', ''))
    words = words.withColumn('word', F.regexp_replace('word', ':', ''))'''
    return words

spark = SparkSession.builder.appName("TwitterSentimentAnalysis").getOrCreate()
# read the tweet data from socket
uschema = StructType([ \
    StructField("Sentiment",IntegerType(),True), \
    StructField("Tweet",StringType(),True)
  ])
lines = spark.readStream.format("socket").option("host", "localhost").option("port", 6100).load()
BRO = lines.select(explode(split(lines.value, ',')))
# words = preprocessing(lines)
# Preprocess the data
query = BRO.writeStream.format('console').start()
query.awaitTermination()


