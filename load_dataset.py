import cx_Oracle
import pandas as pd
import pre_process





ora_conn = cx_Oracle.connect('SDDX1_BASE', '123456', 'localhost:1521/orcl')
df_ora = pd.read_sql('select '
                     'B.DETECT_EQUIP_NO,'
                     'B.POSITION_NO,'
                     'B.BAR_CODE,'
                     'B.LOAD_CURRENT,'
                     'B.PF,'
                     'B.AVE_ERR '
                     'from MT_BASICERR_MET_CONC B, MT_DETECT_TASK D '
                     'where B.DETECT_TASK_NO=M.DETECT_TASK_NO '
#                     'and B.AVE_ERR < 0.6 and B.AVE_ERR >-0.6 '                   
                     'and D.ARRIVE_BATCH_NO=261806084535853', con=ora_conn)
ora_conn.close()


ora_conn = cx_Oracle.connect('SDDX1_BASE', '123456', '222.20.86.79:1521/orcl')
df_ora = pd.read_sql('select '
                     'D.ARRIVE_BATCH_NO,'
                     'B.DETECT_TASK_NO,'
                     'B.DETECT_EQUIP_NO,'
                     'B.POSITION_NO,'
                     'B.BAR_CODE,'
                     'B.LOAD_CURRENT,'
                     'B.PF,'
                     'B.AVE_ERR, '
                     'B.HANDLE_DATE '
                     'from MT_BASICERR_MET_CONC B '
                     'where B.DETECT_TASK_NO=391705153883457 '
                     'or B.DETECT_TASK_NO=391705183920261', con=ora_conn)
ora_conn.close()

load_current = df_ora.LOAD_CURRENT.tolist()
pf = df_ora.PF.tolist()

item_no = pre_process.item_transfer(load_current, pf)

df_ora.insert(5, 'ITEM_NO', item_no)  

df_ora.drop(['LOAD_CURRENT', 'PF'], axis=1, inplace=True)
df_ora.to_csv('E:\Workspaces\Python3\AnomalyDetection\data\dataset.csv', index=False)



dataset = pre_process.data_reform(df_ora)