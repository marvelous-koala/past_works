## 교내 프로젝트, 마코위츠 포트폴리오 이론을 활용한 S&P 500 관련 ETF 투자 포트폴리오 작성
#### 자세한 설명은 첨부한 ppt 참조
##### 마코위츠 포트폴리오
![image](https://user-images.githubusercontent.com/76254564/107885753-584d0f00-6f3f-11eb-8611-16e008853469.png)

### 폴더 설명
#### EDA
##### S&P500 지수와 ETF 선정을 위한 간단한 EDA 진행
##### 운용규모 상위 50위의 ETF를 대상으로 군집분석 진행
![image](https://user-images.githubusercontent.com/76254564/107885517-196a8980-6f3e-11eb-8bf7-27c78343caf9.png)
![image](https://user-images.githubusercontent.com/76254564/107885547-4cad1880-6f3e-11eb-9750-add409cee072.png)
![image](https://user-images.githubusercontent.com/76254564/107885529-30a97700-6f3e-11eb-8b94-298364c02d8c.png)
##### 상관분석과 군집분석으로 ETF를 범주화하여 분류
![image](https://user-images.githubusercontent.com/76254564/107885554-5df62500-6f3e-11eb-9763-24c57aa8dc62.png)
##### 각 범주 별로 2개의 우수한 ETF 선정

#### LSTM 모델
##### 향후 1~5일의 S&P 500 종가를 예측하기 위한 LSTM 모델링, 정확도 향상을 위해 Xgboost 모델과 앙상블 시도

  pyplot.plot(history.history['loss'], label='train')
  pyplot.plot(history.history['val_loss'], label='test')
  pyplot.legend()
  pyplot.show()
  
![image](https://user-images.githubusercontent.com/76254564/107885637-cd6c1480-6f3e-11eb-9691-405c439bf4ad.png)
![image](https://user-images.githubusercontent.com/76254564/107885643-dc52c700-6f3e-11eb-8d0e-372f5f200d37.png)
![image](https://user-images.githubusercontent.com/76254564/107885615-bb8a7180-6f3e-11eb-8e54-be0b8bf2fc22.png)

  def modelfit(alg, train, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

      if useTrainCV:
          xgb_param = param
          xgtrain = xgb.DMatrix(train[predictors].values, label=train[target].values)
          cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=param['n_estimators'], nfold=cv_folds,
              metrics='merror', early_stopping_rounds=early_stopping_rounds)
          xgb_param['n_estimators']=cvresult.shape[0]
          alg.set_params(**xgb_param)

      #Fit the algorithm on the data
      alg.fit(train[predictors], train[target],eval_metric='auc')

      #Predict training set:
      train_predictions = alg.predict(train[predictors])
      train_predprob = alg.predict_proba(train[predictors])[:,1]

      #Print model report:
      print("\nModel Report")
      print(cvresult.shape[0])
      print("Accuracy : %.4g" % metrics.accuracy_score(train[target].values, train_predictions))
      # print("AUC Score (Train): %f" % metrics.roc_auc_score(train[target], train_predprob))

      feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
      feat_imp.plot(kind='bar', title='Feature Importances')
      plt.ylabel('Feature Importance Score')
      
#### Genetic Algorithm
##### 마코위츠 포트폴리오(공분산의 최소화)와 예측모형의 향후 종가를 종합한 최적의 ETF 투자비율 산출
##### NLP(Non - linear programming)를 사용하기 때문에 heuristic 기법(유전 알고리즘)을 활용

##### 공분산 계산기(마코위츠 방법론)
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
      
##### 유전 알고리즘
  def ga_optimizer(ini_date, days):

      algorithm_param = {'max_num_iteration': 2,
                     'population_size':100,
                     'mutation_probability':0.1,
                     'elit_ratio': 0.01,
                     'crossover_probability': 0.5,
                     'parents_portion': 0.3,
                     'crossover_type':'uniform',
                     'max_iteration_without_improv':100}

      varbound = np.array([[0,0.3]]*14)
      ini_date = ini_date


      model=ga(function=f,dimension=14,variable_type='real',variable_boundaries=varbound, algorithm_parameters=algorithm_param)

      model.run()

      result = model.output_dict['variable']

      opt = pd.DataFrame(result , index=name_list)
      opt.columns = [days]

      print('비율의 합')
      print(result.sum())

      return opt
      
##### 계산시간이 오래 걸리는 관계로 병렬연산 처리 진행
  from multiprocessing import Pool
  while current < limit:
      if __name__ == '__main__':
          with Pool(cores) as p:
              finale = p.map(mjh, mulit_list)

      mulit_list = []
      for i in range(0, cores):
          new_series = pd.Series(finale[i])
          fin_df = pd.concat([fin_df, new_series], axis=1)
          mulit_array[i]= mulit_array[i] + cores
          mulit_list.append(mulit_array[i].tolist()[0])
          mulit_list[i]= [int(i) for i in mulit_list[i]]
      current += 1
      print("{0}/{1}".format(current,limit))
      
 ##### simple_trading_bot 폴더의 간단한 트레이딩 알고리즘에 결과를 더하여 매주 리벨런싱이 이루어지는 모델 코딩, 2020년 모의투자로 수익률 비교
 ##### 전통적인 마코위츠 모델에 비해 0.5%의 추가 수익률 달성
 ![image](https://user-images.githubusercontent.com/76254564/107885893-1bcde300-6f40-11eb-8503-529910fcefbe.png)
