# 2020년 빅 콘테스트 혁신 아이디어 분야 운영위원장 상 수상
### 데이터 보안 약관 준수로, input, output data 제거, 공개 가능한 부분만 업로드 되었습니다.
### 5인 1조로 진행하였으며, 직접 작성한 코드만 공유함.


## 파일 설명

### 유동인구 폴더
##### 주어진 자료 시각화
![image](https://user-images.githubusercontent.com/76254564/107884869-89771080-6f3a-11eb-8e58-f7c808b615af.png)
##### 19년과 20년 T 검정 등의 EDA 진행
##### RNN을 활용한 유동인구 예측 시도
##### 공휴일 변수로 인한 예측의 어려움 발생 _ Facebook의 Prophet Library를 활용하여 Holiday variables에 공휴일 변수 입력 & 유동인구 예측 진행
![image](https://user-images.githubusercontent.com/76254564/107885222-5afa3500-6f3c-11eb-8291-5306d3bc8d08.png)

### 군집분석 폴더
##### 주어진 대구와 서울 각 동간의 유사도를 비교하기 위한 군집분석 진행
##### Random search를 응용해 군집분석의 핵심변수 파악 시도

    features = []
    temp_list = []
    ks = range(1,10)
    inertias = []

    for i in range(0,반복):
        # 총 53개의 컬럼 중에 몇개나 뽑을지 무작위로 정함(randrange) = 몇개 뽑을지 picks에 저장
        picks = random.randrange(1,53)

        # picks에 5개가 나오면 5개, 10개가 나오면 10개, 기존의 데이터 칼럼 리스트에서 뽑는다.
        sampling = random.choices(target, k = picks)
        sampling = set(sampling)
        sampling = list(sampling)

        # 예를들어 인구밀도, 성비를 컬럼에서 뽑았다면 이에 해당하는 컬럼들만 모아서 데이터 프레임을 만든다.
        target_data = 군집[sampling]
        data_standadized = StandardScaler().fit_transform(target_data)

        for k in ks:
            model = KMeans(n_clusters=k)
            model.fit(data_standadized)
            inertias.append(model.inertia_)

        plt.plot(ks, inertias, '-o')
        plt.xlabel('number of clusters(K)')
        plt.ylabel('SSE')
        plt.xticks(ks)
        plt.show()
        inertias = []


### EDA 폴더
##### 택배 데이터, 카드 매출, 편의점 매출 등의 데이터 EDA 진행
![image](https://user-images.githubusercontent.com/76254564/107885061-5d0fc400-6f3b-11eb-8837-7e7127f66173.png)
##### 품목 별 상관관계(heatmap), 19년과 20년 차이검정(T 검정, ANOVA 검정), 정규성 검정, 결측치 파악, 이상치 파악, 코로나 확진자와 상관계수, 코로나 검색량과 상관계수 등 검정
![image](https://user-images.githubusercontent.com/76254564/107885096-a52ee680-6f3b-11eb-814f-6965e7715c5b.png)

### Transfer_entropy(TE)
##### TE 개념과 정보의 흐름 개념을 유동인구에 적용, 정보흐름의 중심지역을 파악하고 19년과 20년의 흐름 비교
##### 자세한 설명은 PPT 참조
##### Python을 통한 ADF Test와 R을 통한 AIC Test 진행
##### 19년과 20년 유동인구 정보흐름을 시각화
![image](https://user-images.githubusercontent.com/76254564/107885368-41a5b880-6f3d-11eb-84ea-1e1fc3cf43ec.png)

### Pred_models 폴더
##### 각종 예측 모델
##### 6개국의 신종플루, 메르스, 인플루엔자 데이터를 학습하여 코로나 확진자 예측
    model = Sequential()
    model.add(LSTM(32, input_shape=(look_back, 1)))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=1000, batch_size=240, verbose=0)
##### 다층 모델 및, input_nations를 변경하며 최적의 예측모델 생성(RMSE 기준)
![image](https://user-images.githubusercontent.com/76254564/107885178-1a022080-6f3c-11eb-8e2e-333c4cf3e408.png)
![image](https://user-images.githubusercontent.com/76254564/107885196-3736ef00-6f3c-11eb-8086-a038f02fbf41.png)

### SEIHR_modeling
##### 전염병 확산모델(SEIHR)을 활용한 마스크 착용률 예측
##### SEIHR 모델
![image](https://user-images.githubusercontent.com/76254564/107885273-9eed3a00-6f3c-11eb-8d04-282d268a9e66.png)

    # 1인당 옮기는 감염자 수
    ########################### 질병관리본부 정은경 본부장 "2.5~3 사이라고 본다" (4월 10일) #############################################
    감염률 = 2.75

    # 초기 감염자 수(가정)
    감염자 = 1
    감염자_지역 = 0

    # 초기 잠복환자 수
    잠복환자 = 0
    잠복환자_지역 = 1

    # 지금까지 다 걸린 사람
    총감염자 = 감염자 + 잠복환자
    총감염자_지역 = 감염자_지역 + 잠복환자_지역

    # 초기 격리환자 수
    격리환자 = 0
    격리환자_지역 = 0

    # 원하는 지역의 총 인구(대한민국으로 가정)
    총인구 = 51640000
    # 명동의 월별 평균 유동인구
    지역_인구 = 320000

    # 현재 병에 걸릴 위험이 있는 사람들의 수
    건강인 = 총인구 - 감염자 - 잠복환자
    건강인_지역 = 지역_인구 - 감염자_지역 - 잠복환자_지역

    # 질병에 대한 두려움으로 생활패턴이 바뀐 건강한 사람의 수
    무서운_건강인 = 0
    무서운_건강인_지역 = 0

    #############################사회에 공포가 퍼져나가는 정도, 대구가 4.08이었으므로 조금 낮게 가정해보았음 ################
    공포감 = 4  
    # 대충 명동은 2.4배라고 가정해보자
    공포감_지역 = 10

    # 환자가 n 명일 때 건강인은 무서운_건강인으로 전환됨
    # 논문에서는 환자 1000명 발생 할 때 마다, 한 단위 증가한다고 가정
    전환률 = 1/1000

    ############################### 델타 값 추론하기 ############################### (안되면 1/50 그냥 넣기)

    # 전염병의 전염률을 낮추는 각종 요인들의 합
    # 여기서는 사회적 거리두기로 통칭
    # 거리두기 : WHO 논문에서 마스크를 쓰면 4.92배 감소, 1m 이상 떨어지면 5.61배 감소
    # 12,000명 중에 대한민국 정부가 사전에 발견한 잠복기 환자는 약 4000명 따라서 약 1.5배 감소 효과가 있었을 것
    # 4.92 * 5.61 * 1.5 = 41.4018 약 41.4로 가정
    거리두기 = 1/41.4
    거리두기_지역 = 1/20

    ################################################################################

    # 병에 걸렸다고 회복되는 비율(투병기간 약 14일 이므로, 단순하게 1/14로 가정[논문])
    회복률 = 1/14

    # 코로나 바이러스 평균 잠복기간
    # 질병관리본부 3월 발표자료 인용 (추가 조사 밑 반영되면 좋음)
    잠복기간 = 4.1

    # 코로나 증상 발현 후, 진단/격리까지 소요되는 시간
    진단판정 = 4.6

    # 회복한 환자
    회복환자 = 0

    scatt = []
    mask = []

    for i in range(0,120):
        scatt.append([i,감염자])
        mask.append(무서운_건강인)
        건강인_df = non_fear(건강인, 감염자, 총인구, 공포감)
        무서운_건강인_df = fear(건강인, 무서운_건강인, 감염자, 총인구, 공포감, 거리두기)
        잠복환자_df = incubation(건강인, 무서운_건강인, 감염자, 총인구, 거리두기, 잠복환자)
        감염자_df = infected(잠복환자, 감염자)
        격리환자_df = quarantine(감염자)
        회복환자_df = recoverd()

        건강인 += 건강인_df
        무서운_건강인 += 무서운_건강인_df
        잠복환자 += 잠복환자_df
        감염자 += 감염자_df
        격리환자 += 격리환자_df
        회복환자 += 회복환자_df
        총감염자 += 감염자
        if i >= 9:
            건강인_지역_df = non_fear(건강인_지역, 감염자_지역, 지역_인구, 공포감_지역)
            무서운_건강인_지역_df = fear(건강인_지역, 무서운_건강인_지역, 감염자_지역, 지역_인구, 공포감_지역, 거리두기_지역)
            잠복환자_지역_df = incubation(건강인_지역, 무서운_건강인_지역, 감염자_지역, 지역_인구, 거리두기_지역, 잠복환자_지역)
            감염자_df = infected(잠복환자_지역, 감염자_지역)

            건강인_지역 += 건강인_지역_df
            무서운_건강인_지역 += 무서운_건강인_지역_df
            잠복환자_지역 += 잠복환자_지역_df
            감염자_지역 += 감염자_df
        print(감염자)    
    print('-------------------------------------------------------------------')
    print(총감염자)
    
##### 모델링 결과와 실제 추이 비교
![image](https://user-images.githubusercontent.com/76254564/107885316-d9ef6d80-6f3c-11eb-9f33-353df33259e2.png)
