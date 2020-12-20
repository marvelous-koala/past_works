#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 라이브러리 불러오기

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


# In[2]:


# 파일 소환

import os
path_dir = "./Preds/"
file_list = os.listdir(path_dir)


# In[3]:


file_list

# 레버리지 UWM, TYO, QLD, SDOW 4번째, 5번째, 7번째, 8번째


# In[4]:


# 보기 편하게 이름만 남기기

name_list = []

for i in range(0, len(file_list)):
    name = file_list[i].split('_')[0]
    name_list.append(name)
name_list


# In[5]:


# ETF 불러오기
etfs = [[] for i in range(0,len(file_list))]

for i in range(0, len(file_list)):
    file_name = path_dir + '/' + file_list[i]
    etfs[i] = pd.read_excel(file_name, delimiter=',')
    etfs[i].columns = [name_list[i]+' 날짜', name_list[i] ,'예측가']
    etfs[i][name_list[i]+' 날짜'] = pd.to_datetime(etfs[i][name_list[i]+' 날짜'])
    etfs[i].set_index(name_list[i]+' 날짜', inplace = True)


# In[6]:


# 수익률 계산을 위한 SP500 소환
sp500 = pd.read_csv('./sp500.csv')
sp500['Date'] = pd.to_datetime(sp500['Date'])
sp500.set_index('Date', inplace=True)


# In[7]:


############ 핵심변수, 계산의 기준이 되는 날짜 선정 ############

target_day = date(2020,1,3)


# In[8]:


############## 기준 날짜를 데이터 프레임 상의 서열("위에서 n 번째")로 전환
# 날짜변수 ETF 14개 + SP500 총 15개의 서열 변수로 전환

ini_date = np.array([target_day]*(len(file_list)+1))

# ETF 서열 변수
for k in range(0, len(file_list)):
    ini_date[k] = etfs[k].index.get_loc(ini_date[k].isoformat())
    
# SP500 서열변수
    
target_number = sp500.index.get_loc(target_day.isoformat())
ini_date[-1] = target_number

ini_date


# In[9]:


# 돌아가는지 보기위한 Sample 비율 생성
# 14 종목에 균등하게 투자

ratio = [[1/len(name_list)]*len(name_list)]
test_dict = dict(zip(name_list, ratio[0]))
test_x = pd.DataFrame(test_dict.items(), columns=['name', 'ratio'])
test_x.set_index('name', inplace=True)
test_x = test_x.transpose()
list_x = list(test_x.values[0])
test_x


# In[10]:


## 공분산 계산기
def cov_maker(ratio, ini_date):
    df = pd.DataFrame()

    for k in range(0, len(file_list)):
        
        # 원하는 날짜의 위치를 불러온다
        cov_end = ini_date[k]

        # 20일 간의 공분산이므로 날짜에서 20일 뺴서 [20일 전 ~ 어제까지 범위가 되게 한다]
        cov_start = cov_end - 20

        cov_test = etfs[k].iloc[cov_start : cov_end, :]
        
        # 실제 가격을 기준으로 한 공분산이므로 '예측가'는 drop으로 치워준다
        cov_test = cov_test.drop('예측가', axis=1)
        df = pd.concat([df, cov_test], axis =1)

    coval = df.cov()
    covs1 = test_x.dot(coval)
    cova2 = covs1.dot(ratio.transpose())
    return cova2.values[0][0]


# In[12]:


# 돌아가는지 확인한다.
# 현재 Target_day가 20년 1월 3일이므로, 1월 2일까지 20일간의 데이터를 불러와 공분산행렬 생성
cov_maker(test_x, ini_date)


# In[13]:


## ETF 예상 수익률 계산기
def profit_maker(ratio, ini_date):
    profits = []
    
    # 오늘 실제 가격은 0번 column에 내일 예측가는 1번 컬럼에 있다.
    for k in range(0, len(file_list)):
        current_pri = etfs[k].iloc[ini_date[k],0]
        pred_pri = etfs[k].iloc[ini_date[k]+1,1]    
        # [(내일 예측가격) - (실제가격)] / (실제가격) == 수익률!
        profit = (pred_pri - current_pri)/current_pri
        profits.append(profit)
    
    # 종목별로 수익률 구해서 Dataframe에 예쁘게 정리
    profit_db = pd.DataFrame(profits, index=name_list)
    
    # 14개 종목 * 14개 종목 별 비율 곱해서 최종 합산 수익률
    exp_profit = ratio.dot(profit_db)
    exp_profit = exp_profit[0][0]
    return exp_profit


# In[14]:


# 돌아가는지 확인한다.
# 1월 3일이 Target_date이므로 1월 2일

profit_maker(test_x, ini_date)


# In[15]:


### SP500 수익률 예상하기
# 지난 2일의 수익률이 필요하므로 지난 3일의 자료를 target_length로 설정

def sp500_maker(ini_date):
    target_length = 3
    sp500_profit = []

    # 1월 2일까지의 수익률 예측이므로 12월 30일, 12월 13일, 1월 2일의 가격 불러오기
    sp_data = sp500.iloc[ini_date[-1]-target_length : ini_date[-1],:]
    
    for i in range(0, target_length-1):
        
        # 30일과 31일의 차이 = 수익률1 /// 31일과 2일의 차이 = 수익률2
        incre = (sp_data.iloc[i+1,0]-sp_data.iloc[i,0])/sp_data.iloc[i,0]
        sp500_profit.append(incre)

    # 수익률 이므로 평균
    avg = sum(sp500_profit)/len(sp500_profit)
    return(avg)


# In[18]:


# 페널티 설정
# 페널티 1, 14개 종목의 합이 1이 넘거나 적으면 안됨.

def pen_1(X):
    pen = 0
    
    # 1을 넘거나 안되면 페널티 "1" 부여
    if np.sum(X) > 1.0001:
        pen += 10
    elif np.sum(X) < 0.9999:
        pen += 10
    
    return pen


# In[19]:


# 페널티 2, 레버리지 4 종목의 합은 10%를 넘을 수 없다.

# 레버리지는 ratio 리스트의 4번째, 5번째, 7번째, 8번째에 위치

def pen_2(ratio):
    
    pen = 0
    target = ratio[3] + ratio[4] + ratio[6] + ratio[7]
    
    # 0.15 = 10%를 넘으면 페널티 부여
    if 0.2 >= target >= 0.10:
        pen += 1
    elif target > 0.2:
        pen += 3
    return pen 


# In[20]:


# 페널티 3, 수익률은 상승장에서 SP500의 1.2배 이상, 하락장에서 0.8배 손실까지 허용

def pen_3(ratio, ini_date):
    
    etf_profit = profit_maker(ratio, ini_date)
    pen = 0
    
    # 상승장의 경우
    if sp500_profit >= 0 and sp500_profit * 1.2 > etf_profit:
        pen += 2
    
    # 하락장의 경우
    elif sp500_profit < 0 and sp500_profit * 0.8 < etf_profit:
        pen += 2
    return pen


# In[21]:


############################## 목적함수 설정 ##############################

# 유전알고리즘(GA)는 여기서 나오는 값을 "최소화"하려고 함
# 공분산의 최소화 이므로 주된 목적함수는 공분산이며, 여기에 제약조건을 지키지 못하면 페널티 부여
# 값을 최소화 하려 하므로 페널티를 피하는 방향으로 최적화가 진행됨


def f(X, ini_date):
    
    object = 0
    pen1 = 0
    pen2 = 0
    pen3 = 0
      
    # 14개 비율을 일단 리스트로 만들기
    x_list = X
    
    # 공분산 계산을 위해 데이터 프레임으로 전환
    test_x = pd.DataFrame(x_list , index=name_list).transpose()

    # 페널티 1 + 2 + 3
    pen1 += pen_1(X)
    pen2 += pen_2(x_list)
    pen3 += pen_3(test_x, ini_date)
    
    # 공분산
    object = cov_maker(test_x, ini_date)
    
    ######### 공분산 + 페널티 1 + 페널티 2 + 페널티 3의 값을 반환 (== 목적함수) #######
    return object + pen1 + pen2 + pen3


# In[22]:


def ga_optimizer():
    
    algorithm_param = {'max_num_iteration': 500,
                   'population_size':100,
                   'mutation_probability':0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv':100}

    varbound = np.array([[0,0.3]]*14)
    


    model=ga(function=f,dimension=14,variable_type='real',variable_boundaries=varbound, algorithm_parameters=algorithm_param)

    model.run()

    result = model.output_dict['variable']

    opt = pd.DataFrame(result , index=name_list)
    opt.columns = [days]

    print('비율의 합')
    print(result.sum())
    
    return opt


# In[23]:


def run_ga(ini_date):

    result_df = pd.DataFrame()
    days = 0

    while days < per_core:

        print('{0}일 중 {1}일 진행중'.format(until, days+1))
        sp500_profit = sp500_maker(ini_date)    
        ga_df = ga_optimizer()
        result_df = pd.concat([result_df, ga_df], axis=1)


        ini_date += 1
        days += 1




# In[ ]:


import time, os
from multiprocessing import Pool, Process, Queue


# In[ ]:


import multiprocessing as mp
cores = mp.cpu_count()

ini_multi = [ini_date].copy()*cores

mulit_array = []
until = 16
per_core = until/cores

for i in range(0, cores):
    ini = np.array([ini_multi[i]]) + int(until/cores * i)
    mulit_array.append(ini)

numbers = mulit_array
procs = []

if __name__ == '__main__':


    for index, number in enumerate(numbers):
        ini_data = np.array(number[0])
        proc = Process(target=run_ga, args=(ini_data)
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

