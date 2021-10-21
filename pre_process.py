import traceback
import numpy as np
from scipy import stats

def item_transfer(load_current, pf):
    '''
    对数据类型进行预处理
    :param data:
    :return:
    '''
    try:
        item_no = np.zeros(len(load_current))
        for i in range(len(load_current)):
            if load_current[i] == '05' and pf[i] == '01':
                item_no[i] = 1
            elif load_current[i] == '05' and pf[i] == '07':
                item_no[i] = 2
            elif load_current[i] == '07' and pf[i] == '07':
                item_no[i] = 3
            elif load_current[i] == '08' and pf[i] == '01':
                item_no[i] = 4
            elif load_current[i] == '08' and pf[i] == '07':
                item_no[i] = 5
            elif load_current[i] == '09' and pf[i] == '01':
                item_no[i] = 6
            elif load_current[i] == '00' and pf[i] == '01':
                item_no[i] = 7
            elif load_current[i] == '00' and pf[i] == '07':
                item_no[i] = 8
            elif load_current[i] == '01' and pf[i] == '01':
                item_no[i] = 9
            elif load_current[i] == '01' and pf[i] == '07':
                item_no[i] = 10
            else:
                item_no[i] = 0
    except:
        item_no = np.zeros(len(load_current))
        traceback.print_exc()
    return item_no

def data_reform(df_ora):
    try:
        equip_no = np.array(list(map(eval, df_ora.DETECT_EQUIP_NO.tolist())))
        position_no = np.array(list(map(eval, df_ora.POSITION_NO.tolist())))

        barcode = np.array(df_ora.BAR_CODE.tolist())

        error = np.array(list(map(eval, df_ora.AVE_ERR.tolist())))

        load_current = df_ora.LOAD_CURRENT.tolist()
        pf = df_ora.PF.tolist()

        item_no = item_transfer(load_current, pf)

        dataset = np.zeros((20, 60, 60))
        dataset_qualified = np.zeros((20, 60, 60))

        for i in range(20):
            equip_index = np.argwhere(equip_no == i + 1)
            for j in range(60):
                postion_index = np.intersect1d(equip_index, np.argwhere(position_no == j + 1))
                for k in range(10):
                    item_index = np.intersect1d(postion_index, np.argwhere(item_no == k + 1))
                    item_error = error[item_index]
#                    item_error = np.random.normal(0,1,30)
#                    r = round(len(item_error)*0.1)
#                    
#                    item_error = np.array(sorted(item_error, key=abs)[:-r])
                    
                    if len(item_error) in [0,1]:
                        item_error = np.array([np.nan])
                    
#                    item_error_qualified = [_ for _ in item_error if _<0.6 and _>-0.6]
#
#                    if len(item_error_qualified) in [0,1]:
#                        item_error_qualified = np.array([np.nan])
                    
                    # method 1----------------------------
                    
                    mu = np.mean(item_error)
                    maximum = np.max(item_error)
                    minimum = np.min(item_error)
                    var = np.var(item_error)
                    skew = stats.skew(item_error)
                    kurtosis = stats.kurtosis(item_error)
#                    unqualified_rate = (np.sum(item_error>0.6) + np.sum(item_error<-0.6))/item_error.size
#                    unqualified_rate = int(np.sum(item_error_qualified>0.6) + np.sum(item_error_qualified<-0.6))/item_error_qualified.size
#                    unqualified_rate = 0
                    
                    dataset[i][j][k] = minimum
                    dataset[i][j][k + 10] = maximum
                    dataset[i][j][k + 20] = mu
                    dataset[i][j][k + 30] = var
                    dataset[i][j][k + 40] = skew
                    dataset[i][j][k + 50] = kurtosis
                    
#                    item_error_qualified = [_ for _ in item_error if _<0.6 and _>-0.6]
#                    
#                                
#                    if len(item_error_qualified) in [0,1]:
#                        item_error_qualified = np.array([np.nan])
#                    
#                    mu = np.mean(item_error_qualified)
#                    maximum = np.max(item_error_qualified)
#                    minimum = np.min(item_error_qualified)
#                    var = np.var(item_error_qualified)
#                    skew = stats.skew(item_error_qualified)
#                    kurtosis = stats.kurtosis(item_error_qualified)
#                    
#                    dataset_qualified[i][j][k] = minimum
#                    dataset_qualified[i][j][k + 10] = maximum
#                    dataset_qualified[i][j][k + 20] = mu
#                    dataset_qualified[i][j][k + 30] = var
#                    dataset_qualified[i][j][k + 40] = skew
#                    dataset_qualified[i][j][k + 50] = kurtosis
                    
#                    dataset[i][j][k + 60] = unqualified_rate
                    # --------------------------------------
#                    # method 2----------------------------
#                    
#                    des = stats.describe(item_error)
#                    
#                    dataset[i][j][k] = des[1][0]
#                    dataset[i][j][k + 10] = des[1][1]
#                    dataset[i][j][k + 20] = des[2]
#                    dataset[i][j][k + 30] = des[3]
#                    dataset[i][j][k + 40] = des[4]
#                    dataset[i][j][k + 50] = des[5]
#                    # --------------------------------------
                    
                    
    except:
        dataset = np.zeros((20, 60, 70))
        dataset_qualified = np.zeros((20, 60, 60))
        traceback.print_exc()
    return dataset

def data_reform_qua_rate(df_ora):
    try:
        equip_no = np.array(list(map(eval, df_ora.DETECT_EQUIP_NO.tolist())))
        position_no = np.array(list(map(eval, df_ora.POSITION_NO.tolist())))

        barcode = np.array(df_ora.BAR_CODE.tolist())

        err_conclude = np.array(df_ora.BASICERR_CONC_CODE.tolist())


        dataset = np.zeros((60, 20))
        for i in range(20):
            equip_index = np.argwhere(equip_no == i + 1)
            for j in range(60):
                postion_index = np.intersect1d(equip_index, np.argwhere(position_no == j + 1))
                conclude = err_conclude[postion_index]
                unqua_con = np.sum(conclude != '01')
                dataset[j][i] = unqua_con/len(conclude)
                
                    
    except:
        dataset = np.zeros((60, 20))
        traceback.print_exc()
    return dataset

def get_raw_number(df_ora):
    try:
        equip_no = np.array(list(map(eval, df_ora.DETECT_EQUIP_NO.tolist())))
        position_no = np.array(list(map(eval, df_ora.POSITION_NO.tolist())))

        barcode = np.array(df_ora.BAR_CODE.tolist())

        error = np.array(list(map(eval, df_ora.AVE_ERR.tolist())))

        load_current = df_ora.LOAD_CURRENT.tolist()
        pf = df_ora.PF.tolist()

        item_no = item_transfer(load_current, pf)


        raw_number = [[[] for _ in range(60)] for _ in range(20)]

        for i in range(20):
            equip_index = np.argwhere(equip_no == i + 1)
            for j in range(60):
                postion_index = np.intersect1d(equip_index, np.argwhere(position_no == j + 1))
                item_error = error[postion_index]
                raw_number[i][j] =  item_error
                
                    
    except:
        raw_number = [[[] for _ in range(60)] for _ in range(20)]
        traceback.print_exc()
    return raw_number
