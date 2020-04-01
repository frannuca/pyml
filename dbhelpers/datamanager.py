import sqlite3
import numpy as np
import pandas as pd
import string
import sqlite3
import json
import requests
import sqlite_utils
from datetime import datetime
from . import table_prototypes
from .table_prototypes import *
from .sqlitehelpers import *
import sys
from os import listdir
from os.path import isfile, join
import re
from pyspark.sql import SparkSession

class RawTableManager(Raw_Prototypes):
    
    def __init__(self,db_file):
        Raw_Prototypes.__init__(self)
        self.conn = create_connection(db_file)
    

    def __enter__(self):
        #ttysetattr etc goes here before opening and returning the file object
        if not self.conn:
            self.__create_connection()
        return self

    def __exit__(self, type, value, traceback):
        #Exception handling here
        if self.conn:
            self.conn.close()

    def close(self):
        self.conn.close()

    def __create_table(self, create_table_sql):
        """ create a table from the create_table_sql statement
        :param conn: Connection object
        :param create_table_sql: a CREATE TABLE statement
        :return:
        """
        conn = self.conn
        try:
            c = conn.cursor()
            c.execute(create_table_sql)
        except Error as e:
            print(e)
        finally:
            c.close()

    def __create_table_from_frame(self,df,table_name):
        conn = self.conn        
        conn.commit()
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.execute(f"delete from {table_name}")
        conn.commit()
        
    def __insert_into_table_from_frame(self,df,table_name):
        conn = self.conn
        df.to_sql(table_name, conn, if_exists='replace', index=False)

    
    def generate_raw_db(self):
        self.runAction(self.__create_table_from_frame)
    
    def getTables(self):
        return [t for t in self.prototypes]

    def generate_from_data(self,folderpath):
        self.generate_raw_db()
        csvtables = set([f +".csv" for f in self.getTables()])
        filesinfolder = set(listdir(folderpath))
        files = csvtables.intersection(filesinfolder)
        for f in files:
            tablename = re.sub(".csv","",f)
            df = pd.read_csv(join(folderpath,f))
            df.to_sql(tablename,self.conn,if_exists='replace')

    def getSparkDF(self,tablename:str,spark:SparkSession):
        pdDF = pd.read_sql_query(f'SELECT * FROM {tablename}', self.conn)          
        df = spark.createDataFrame(pdDF)
        return df