# 2020년 빅 콘테스트 혁신 아이디어 분야 운영위원장 상 수상
### 데이터 보안 약관 준수로, input, output data 제거, 공개 가능한 부분만 업로드 되었습니다.


## 파일 설명

### 유동인구 폴더
##### 주어진 자료 시각화
![image](https://user-images.githubusercontent.com/76254564/107884869-89771080-6f3a-11eb-8e58-f7c808b615af.png)
##### 19년과 20년 T 검정 등의 EDA 진행
##### RNN을 활용한 유동인구 예측 시도
##### 공휴일 변수로 인한 예측의 어려움 발생 _ Facebook의 Prophet Library를 활용하여 Holiday variables에 공휴일 변수 입력 & 유동인구 예측 진행

### 군집분석 폴더
##### 주어진 대구와 서울 각 동간의 유사도를 비교하기 위한 군집분석 진행
##### Random search를 응용해 군집분석의 핵심변수 파악 시도
"""
# 예쁘게 모으는 거에 성공하면, 성공적인 칼럼들이 무엇인지 저장 할 칼럼
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
"""
