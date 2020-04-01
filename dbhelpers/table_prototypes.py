import numpy as np
import pandas as pd
from datetime import datetime


class Raw_Prototypes:
    def __init__(self):
        
        self.prototypes={
            
            "raw_mbis_kpi_metadata":lambda : pd.DataFrame(data=[["X","X",datetime(1900,1,1),datetime(1900,1,1),datetime(1900,1,1),"X",datetime(1900,1,1)]], \
                        columns=["kpi_key","desc_short","valid_from","valid_to","update_date","update_user","h_load_datetime"]),

            "raw_mbis_kri_metadata":lambda : pd.DataFrame(data=[["X","X","X","X",datetime(1900,1,1),datetime(1900,1,1),datetime(1900,1,1),"X",datetime(1900,1,1)]], \
                        columns=["kri_key","kri_id","desc_short","desc_long","valid_from","valid_to","update_date","update_user","h_load_datetime"])

        }

    
    
    def runAction(self,action):
        for tablename in self.prototypes:
            action(self.prototypes[tablename](),tablename)