import numpy as np
import pandas as pd
from datetime import timedelta, datetime

# knn 적용
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 참고로 이 모델은 dataframe이 들어가야 하므로, csv파일을 수정할 필요가 있음. 사용자 입장을 생각하지 않은 모델이므로 수정과 수정을 거듭
class Model :
    data_list = [] # reindex를 하기 위한 리스트

    # 생성자 선택 최대 5가지 데이터를 넣을 수 있음, 최소는 2개 아니면 에러뜸
    def __init__ (self, dataframe1=None, dataframe2=None, dataframe3=None, dataframe4=None, dataframe5=None, wind_dataframe1=None, wind_dataframe2=None, wind_dataframe3=None, wind_dataframe4=None, wind_dataframe5=None) :
        self.dataframe1 = dataframe1
        self.dataframe2 = dataframe2
        self.dataframe3 = dataframe3
        self.dataframe4 = dataframe4
        self.dataframe5 = dataframe5
        self.wind_dataframe1 = wind_dataframe1
        self.wind_dataframe2 = wind_dataframe2
        self.wind_dataframe3 = wind_dataframe3
        self.wind_dataframe4 = wind_dataframe4
        self.wind_dataframe5 = wind_dataframe5

    @staticmethod
    # outliar 제거 함수 (IQR 설정)
    def remove_out(*dataframes, remove_col) :
        for i, dataframe in enumerate(dataframes) :
            for k in remove_col :
                level_1q = dataframe[k].quantile(0.25)
                level_3q = dataframe[k].quantile(0.75)
                IQR = level_3q - level_1q
                rev_range = 10

                outliar_h = dataframe[k] >= level_3q + (rev_range * IQR)
                outliar_l = dataframe[k] <= level_1q - (rev_range * IQR)

                a=dataframe[outliar_h].index
                b=dataframe[outliar_l].index

                dataframe.drop(a, inplace=True)
                dataframe.drop(b, inplace=True)

    @staticmethod   
    # 미세먼지 농도 라벨 설정
    def air_quality_label(pm25):
        if pm25 <= 35:
            return '좋음,보통'
        else:
            return '나쁨,매우 나쁨'
    
    @staticmethod
    # 풍향, 풍속 데이터 삽입
    def wind_arim(*dataframe) :
        Flag = False
        while Flag == False :
            c = 0
            for data in dataframe :
                data.name = input("지역을 입력하세요. => ") # 이건 수정과정이 필요할 것
                c += 1
                data['일시'] = data['일시'].apply(pd.to_datetime)
                data_avg = data.groupby([pd.Grouper(key='일시', freq='H')]).mean()
                if data.name == '상봉동' :
                    for data2 in Model.data_list :
                        if data2['name'] == 1 :
                            data2['data']['wind_dir'] = data_avg['풍향(16방위)']
                            data2['data']['wind_speed'] = data_avg['풍속(m/s)']
                elif data.name == "상대동" :
                    continue
                elif data.name == "정촌면" :
                    for data2 in Model.data_list :
                        if data2['name'] == 0 :
                            data2['data']['wind_dir'] = data_avg['풍향(16방위)']
                            data2['data']['wind_speed'] = data_avg['풍속(m/s)']
                elif data.name == "대안동" :
                    for data2 in Model.data_list :
                        if data2['name'] == 2 :
                            data2['data']['temp'] = data_avg['기온']
                            data2['data']['humi'] = data_avg['상대습도(%)']
                            data2['data']['wind_dir'] = data_avg['풍향(16방위)']
                            data2['data']['wind_speed'] = data_avg['풍속(m/s)']
                else :
                    print("현재 설정되지 않은 지역입니다.")
                    continue
            if c == len(Model.data_list) :
                Flag = True

    @staticmethod
    # 풍향과 풍속을 풍벡터로 표현
    def wind_vector(*dataframe):
        for data in dataframe :
            # 풍향을 라디안으로 변환
            theta = np.deg2rad(data.wind_dir)

            # 극 좌표에서 직교 좌표로 변환
            x = data.wind_speed * np.cos(theta)
            y = data.wind_speed * np.sin(theta)

            data['wind_vector_x'] = x
            data['wind_vector_y'] = y
            data.drop('wind_dir', axis=1, inplace=True)
            data.drop('wind_speed', axis=1, inplace=True)
    
    @staticmethod
    # knn 적용
    def knn(data_list) :    
        X_train = pd.concat([data_list[0]['data'], data_list[1]['data']])
        X_train = X_train.drop(columns=['pm10','pm25' ,'air_quality_label'])
        y_train = pd.concat([data_list[0]['data']['air_quality_label'], data_list[1]['data']['air_quality_label']]) # 현재는 2개만 설정

        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # knn 적용
        k = 5  # 이웃의 수를 5로 설정
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        # 예측을 수행 할 대상
        X = data_list[2]['data'].drop(columns=['pm10','pm25', 'air_quality_label'])

        y_pred = knn.predict(X) # 예측 수행
        y_pred_proba = knn.predict_proba(X) # 각각의 데이터 포인트가 각 클래스에 속할 확률을 나타내는 2차원 배열

        # 예측한 결과 출력
        result = pd.DataFrame({'air_quality_label_pred': y_pred})
        result.index = X.index
        result['probability'] = np.max(y_pred_proba, axis=1)
        pd.display(result)

        y = data_list[2]['data']['air_quality_label'].tolist() # 비교할 대상

        result['air_quality_label'] = y
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            pd.display(result)

        # arim과 air이 같을때를 따로 찾아보기
        mask = result['air_quality_label'] == result['air_quality_label_pred']
        result_1 = result[mask]
        result_1.dropna(inplace=True)

        pd.display(result_1)

        # arim과 air이 다를때를 따로 찾아보기
        mask = result['air_quality_label'] != result['air_quality_label_pred']
        result_2 = result[mask]
        result_2.dropna(inplace=True)

        result_1.info()
        result_2.info()

    # arim 데이터 reindex과정까지 규칙 진주성 = 1, 학교 = 0, 대안동 = 2
    def model_1(*dataframe, wind_dataframe1, wind_dataframe2, wind_dataframe3, wind_dataframe4, wind_dataframe5) :
        count = 0 # 데이터 개수
        Flag = False # 데이터 개수가 달라 다시 시작
        while Flag == False :
            for data in dataframe : # 데이터들의 전처리 과정
                if data == None :
                    continue
                data.name = input("지역을 입력하세요. => ") # 이건 수정과정이 필요할 것
                if data.name == "학교" or data.name == "진주성" :
                    print("#####\ndata_name =",data.name,"\n#####")
                    # index를 datetime 형식으로 바꿔주기
                    data['reg_date'] = data.reg_date.apply(pd.to_datetime)
                    # 이름 설정
                    data.name = int(input("지역 이름을 설정해 주세요 ex)학교=0,진주성=1,대안동=2 -> "))
                    # 아웃라이어 제거
                    remove_out(data, remove_col=['pm25'])
                    # 1시간 단위로 측정 나누기
                    data_avg = data.groupby([pd.Grouper(key='reg_date', freq='H')]).mean()
                    # arim 데이터 수정
                    data_76 = data_avg.loc[data_avg.loc[data_avg.pm25 >= 76].index]
                    data_76.pm25 -= 10.65
                    # arim 데이터 수정 후 대입
                    data_avg.loc[data_76.index, 'pm25'] = data_76['pm25']
                    # 좋음, 보통, 나쁨, 매우나쁨 카데고리 분류
                    data_avg['air_quality_label'] = data_avg['pm25'].apply(lambda x: air_quality_label(x))
                    # 참고로 Nan 값을 가진 pm2.5 데이터를 소거 해주는 작업이 필요!
                    data_avg = data_avg.dropna(subset=['pm25'], axis=0)
                    # 데이터를 저장
                    Model.data_list.append({'name': data.name, 'data': data_avg})
                    count += 1
                else : # airKorea
                    print("#####\ndata_name =",data.name,"\n#####")
                    # index를 datetime 형식으로 바꿔주기
                    data['date'] = data.date.apply(pd.to_datetime)
                    # 이름 설정
                    data.name = int(input("지역 이름을 설정해 주세요 ex)학교=0,진주성=1,대안동=2 -> "))
                    # 아웃라이어 제거
                    remove_out(data, remove_col=['pm25'])
                    # 1시간 단위로 측정 나누기
                    data_avg = data.groupby([pd.Grouper(key='date', freq='H')]).mean()
                    data_avg.loc[data_76.index, 'pm25'] = data_76['pm25']
                    # 좋음, 보통, 나쁨, 매우나쁨 카데고리 분류
                    data_avg['air_quality_label'] = data_avg['pm25'].apply(lambda x: air_quality_label(x))
                    # 참고로 Nan 값을 가진 pm2.5 데이터를 소거 해주는 작업이 필요!
                    data_avg = data_avg.dropna(subset=['pm25'], axis=0)
                    # 데이터를 저장
                    Model.data_list.append({'name': data.name, 'data': data_avg})
                    count += 1
            # 가장 작은 data의 인덱스와 나머지 데이터들의 인덱스 같게 만들어주기
            if count == 2 :
                idx = set(Model.data_list[0]['data'].index).intersection(
                    set(Model.data_list[1]['data'].index)
                )
                for data in Model.data_list :
                    data['data'] = data['data'].reindex(idx)
                Flag = True # 종료
            elif count == 3 :
                idx = set(Model.data_list[0]['data'].index).intersection(
                    set(Model.data_list[1]['data'].index),
                    set(Model.data_list[2]['data'].index)
                )
                for data in Model.data_list :
                    data['data'] = data['data'].reindex(idx)
                Flag = True # 종료
            elif count == 4 :
                idx = set(Model.data_list[0].index).intersection(
                    set(Model.data_list[1].index),
                    set(Model.data_list[2].index),
                    set(Model.data_list[3].index)
                )
                for data in Model.data_list :
                    data['data'] = data['data'].reindex(idx)
                Flag = True # 종료
            elif count == 5 :
                idx = set(Model.data_list[0].index).intersection(
                    set(Model.data_list[1].index),
                    set(Model.data_list[2].index),
                    set(Model.data_list[3].index),
                    set(Model.data_list[4].index)
                )
                for data in Model.data_list :
                    data['data'] = data['data'].reindex(idx)
                Flag = True # 종료
        if count == 2 :
            wind_arim(wind_dataframe1, wind_dataframe2)
        elif count == 3 :
            wind_arim(wind_dataframe1, wind_dataframe2, wind_dataframe3)
        elif count == 4 :
            wind_arim(wind_dataframe1, wind_dataframe2, wind_dataframe3, wind_dataframe4)
        elif count == 5 :
            wind_arim(wind_dataframe1, wind_dataframe2, wind_dataframe3, wind_dataframe4, wind_dataframe5)

        for data in Model.data_list :
            wind_vector(data['data'])

        # index 정렬이 안되서 출력이 돼 그래프가 이상하게 그려져 적용시킴
        for data in Model.data_list :
            if data['name'] == 0 or data['name'] == 1 :
                data['data'].drop('no', axis=1, inplace=True)
            data['data'].sort_index(inplace=True)
            pd.display(data['data'])
        knn(Model.data_list)

    def execute_model(self) :
        self.model_1(self.dataframe1, self.dataframe2, self.dataframe3, self.dataframe4, self.dataframe5, wind_dataframe1=self.wind_dataframe1, wind_dataframe2=self.wind_dataframe2, wind_dataframe3=self.wind_dataframe3, wind_dataframe4=self.wind_dataframe4, wind_dataframe5=self.wind_dataframe5)
        return(self.knn(Model.data_list))