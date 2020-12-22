import pandas as pd
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import numpy as np

from datetime import datetime  
from datetime import timedelta
from datetime import date 
import math
from geneticalgorithm import geneticalgorithm as ga

import os

import time, os
from multiprocessing import Pool
import multiprocessing as mp
from multiprocessing import Process


path_dir = "./Preds/"
file_list = os.listdir(path_dir)

name_list = []

for i in range(0, len(file_list)):
    name = file_list[i].split('_')[0]
    name_list.append(name)

etfs = [[] for i in range(0,len(file_list))]

for i in range(0, len(file_list)):
    file_name = path_dir + '/' + file_list[i]
    etfs[i] = pd.read_excel(file_name, delimiter=',')
    etfs[i].columns = [name_list[i]+' 날짜', name_list[i] ,'예측가']
    etfs[i][name_list[i]+' 날짜'] = pd.to_datetime(etfs[i][name_list[i]+' 날짜'])
    etfs[i].set_index(name_list[i]+' 날짜', inplace = True)


sp500 = pd.read_csv('./sp500.csv')
sp500['Date'] = pd.to_datetime(sp500['Date'])
sp500.set_index('Date', inplace=True)

## 공분산 계산기
def cov_maker(ratio):
    df = pd.DataFrame()

    for k in range(0, len(file_list)):
        
        # 원하는 날짜의 위치를 불러온다
        cov_end = start_d[k]

        # 20일 간의 공분산이므로 날짜에서 20일 뺴서 [20일 전 ~ 어제까지 범위가 되게 한다]
        cov_start = cov_end - 20

        cov_test = etfs[k].iloc[cov_start : cov_end, :]
        
        # 실제 가격을 기준으로 한 공분산이므로 '예측가'는 drop으로 치워준다
        cov_test = cov_test.drop('예측가', axis=1)
        df = pd.concat([df, cov_test], axis =1)

    coval = df.cov()
    covs1 = ratio.dot(coval)
    cova2 = covs1.dot(ratio.transpose())
    return cova2.values[0][0]




## ETF 예상 수익률 계산기
def profit_maker(ratio):
    profits = []
    
    # 오늘 실제 가격은 0번 column에 내일 예측가는 1번 컬럼에 있다.
    for k in range(0, len(file_list)):
        current_pri = etfs[k].iloc[start_d[k],0]
        pred_pri = etfs[k].iloc[start_d[k]+1,1]    
        # [(내일 예측가격) - (실제가격)] / (실제가격) == 수익률!
        profit = (pred_pri - current_pri)/current_pri
        profits.append(profit)
    
    # 종목별로 수익률 구해서 Dataframe에 예쁘게 정리
    profit_db = pd.DataFrame(profits, index=name_list)
    
    # 14개 종목 * 14개 종목 별 비율 곱해서 최종 합산 수익률
    exp_profit = ratio.dot(profit_db)
    exp_profit = exp_profit[0][0]
    return exp_profit



### SP500 수익률 예상하기
# 지난 2일의 수익률이 필요하므로 지난 3일의 자료를 target_length로 설정

def sp500_maker(X):
    target_length = X
    sp500_profit = []

    # 1월 2일까지의 수익률 예측이므로 12월 30일, 12월 13일, 1월 2일의 가격 불러오기
    sp_data = sp500.iloc[start_d[-1]-target_length : start_d[-1],:]
    
    for i in range(0, target_length-1):
        
        # 30일과 31일의 차이 = 수익률1 /// 31일과 2일의 차이 = 수익률2
        incre = (sp_data.iloc[i+1,0]-sp_data.iloc[i,0])/sp_data.iloc[i,0]
        sp500_profit.append(incre)

    # 수익률 이므로 평균
    avg = sum(sp500_profit)/len(sp500_profit)
    return(avg)


# In[15]:


# 페널티 설정
# 페널티 1, 14개 종목의 합이 1이 넘거나 적으면 안됨.

def pen_1(X):
    pen = 0
    
    # 1을 넘거나 안되면 페널티 "1" 부여
    if np.sum(X) > 1.01:
        pen += np.sum(X)
    elif np.sum(X) < 0.99:
        pen += 1-np.sum(X)
    
    return pen


# In[16]:


# 페널티 2, 레버리지 4 종목의 합은 10%를 넘을 수 없다.

# 레버리지는 ratio 리스트의 4번째, 5번째, 7번째, 8번째에 위치

def pen_2(ratio):
    
    pen = 0
    target = ratio[3] + ratio[4] + ratio[6] + ratio[7]
#    pen += target
    
    # 0.15 = 10%를 넘으면 페널티 부여
    if 0.4 >= target >= 0.3:
        pen += target * 2
    elif target > 0.4:
        pen += target * 4

    return pen 


# In[17]:


# 페널티 3, 수익률은 상승장에서 SP500의 1.2배 이상, 하락장에서 0.8배 손실까지 허용

def pen_3(ratio):
    
    neck = 0.007
    
    etf_profit = profit_maker(ratio)
    sp500_profit = sp500_maker(3)
    pen = 0
    gap = etf_profit - sp500_profit
    
     # 상승장의 경우
    if sp500_profit >= neck and sp500_profit * 1.4 > etf_profit:
        pen += 0.8
    
     # 하락장의 경우
    elif sp500_profit < -neck and sp500_profit * 0.6 < etf_profit:
        pen += 0.4
        
    return pen, gap

def pen_4(ratio):
    pen = 0
    sp500_profit = sp500_maker(3)
#    if sp500_profit > 0 and ratio[4] + ratio[7] > 0.05:
#        pen += (ratio[4] + ratio[7])*8
#    elif sp500_profit < 0 and ratio[3] + ratio[6] > 0.05:
#        pen += (ratio[3] + ratio[6]) *8
        
    bulls = ratio[3] + ratio[6]
    bears = ratio[4] + ratio[7]
    gaps = abs(bulls - bears)
    
    return pen, gaps


# In[18]:


############################## 목적함수 설정 ##############################

# 유전알고리즘(GA)는 여기서 나오는 값을 "최소화"하려고 함
# 공분산의 최소화 이므로 주된 목적함수는 공분산이며, 여기에 제약조건을 지키지 못하면 페널티 부여
# 값을 최소화 하려 하므로 페널티를 피하는 방향으로 최적화가 진행됨


def f(X):
    object = 0
    pen1 = 0
    pen2 = 0
    pen3 = 0
    pen4 = 0
    
    profit_ratio = 0.3
      
    # 14개 비율을 일단 리스트로 만들기
    x_list = list(X)
    
    # 공분산 계산을 위해 데이터 프레임으로 전환
    test_x = pd.DataFrame(x_list , index=name_list).transpose()

    # 페널티 1 + 2 + 3
    pen1 += pen_1(X)
    pen2 += pen_2(x_list)
    penal, gap = pen_3(test_x)
    
    pen3 += penal
    
    pen4_t, bull_n_bear = pen_4(x_list)
    pen4 += pen4_t
    
    # 공분산
    object = cov_maker(test_x)
    
    ######### 공분산 + 페널티 1 + 페널티 2 + 페널티 3의 값을 반환 (== 목적함수) #######
    return object + pen1 + pen2 + pen3 + pen4 - gap*3 - bull_n_bear


# In[19]:


def ga_optimizer(days):
    
    algorithm_param = {'max_num_iteration': None,
                   'population_size':100,
                   'mutation_probability':0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv':None}

    varbound = np.array([[0,0.4]]*14)


    model=ga(function=f,dimension=14,variable_type='real',variable_boundaries=varbound, algorithm_parameters=algorithm_param)

    model.run()

    result = model.output_dict['variable']
    
    return result


# In[20]:


def run_ga():
    global start_d
    days = 0
    until =1
    sp500profit = 0
           
    result = ga_optimizer(days)

    return result


def mjh(ini_date):
    global start_d
    start_d = ini_date
    result = 0
    sp500profit = sp500_maker(3)
    if abs(sp500profit) > 0.007:
        result = run_ga()
    else:
        result = [0.1, 0.2, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1]
            
    return result

