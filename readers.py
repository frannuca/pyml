import findspark
findspark.init()
import os
import sys
import pyspark
import pyspark.sql
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import functions
from pyspark.sql.functions import *
from pyspark.sql.types import *
import sqlite3

sc = SparkContext(appName="MyFirstApp") 
spark = SparkSession(sc) 

basefolder = """/Users/fran/code/bigdata/Spark-The-Definitive-Guide/data/flight-data/jdbc"""

driver = "org.sqlite.JDBC"
path = f'{basefolder}/my-sqlite.db'
url=f'jdbc:sqlite:${path}'
tablename = "flight_info"

dbFrame = spark.read.format("jdbc")
            