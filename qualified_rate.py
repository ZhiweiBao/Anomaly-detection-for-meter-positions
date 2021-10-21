# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 16:52:53 2019

@author: dell
"""

import cx_Oracle
import pandas as pd

import pre_process

ora_conn = cx_Oracle.connect('SDDX1_BASE', '123456', 'localhost:1521/orcl')
df_ora = pd.read_sql('select distinct '
                     'D.DETECT_EQUIP_NO,D.POSITION_NO,B.BAR_CODE,D.BASICERR_CONC_CODE '
                     'from MT_DETECT_MET_RSLT D, MT_MODEL M, MT_BASICERR_MET_CONC B '
                     'where D.DETECT_TASK_NO=M.DETECT_TASK_NO '
                     'and D.BAR_CODE = B.BAR_CODE '
                     'and M.ARRIVE_BATCH_NO=261806084535853', con=ora_conn)
ora_conn.close()


dataset = pre_process.data_reform_qua_rate(df_ora)


