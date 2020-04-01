import dbhelpers
from dbhelpers.datamanager import *

if __name__ == '__main__':    
   dbm = RawTableManager(db_file="./pythonsqlite.db")
   #dbm.generate_from_data("/Users/fran/code/bigdata/pythonsparkbook/spark-warehouse")
   sparkdf = dbm.getSparkDF("raw_mbis_kri_metadata")
   print("finished")