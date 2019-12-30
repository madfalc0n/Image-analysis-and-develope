# 이미지 분석 및 개발 실무 과정 1차 프로젝트

## 1. 주제 : 서울시 행정구역별 범죄발생 빈도 분석

## 2. 기한 : 2020년 1월 03일(금) 까지

## 3. 데이터 출처 정보:

1. [서울시 5대 범죄 발생현황 통계(14년~18년)](https://data.seoul.go.kr/dataList/datasetView.do?infId=316&srvType=C&serviceKind=2)
2. [서울시 자치구 년도별 CCTV 설치 현황(11년~ 18년)](https://data.seoul.go.kr/dataList/datasetView.do?infId=OA-2734&srvType=F&serviceKind=1&currentPageNo=1)
3. [서울시 주민등록인구 (구별) 통계](https://data.seoul.go.kr/dataList/datasetView.do?serviceKind=2&infId=419&srvType=S&stcSrl=419)
4. [서울시 자치구 별 위도,경도](https://github.com/southkorea/seoul-maps/tree/master/juso/2015/json)

## 4. 진행 상황

### 1. 12월 28일

##### 1.1 진행상황

1. 공공 데이터 전처리(CCTV 설치 수, 인구 수, 범죄 발생 수) 및 데이터 시각화(CCTV 와 범죄 발생 수 관계, 인구수 와 범죄 발생 수 관계) 진행

### 2. 12월 29일 ~ 30일

##### 2.1 진행상황

1. 데이터 시각화를 위한 geopandas 모듈 숙지
2. 서울시 자치구 별 위도,경도 데이터 확인(위 데이터 참고)

3. 강사님 피드백을 통해 분석방향 및 데이터 시각화 방향 제시
   1. 지역별 10만 인구당 CCTV 설치 수 및 범죄 발생 수 
   2. 각 지역별 해외인구 수에 따른 범죄수
   3. 지역별 숙박업소 및 식품점(음식점, 유흥업소 등) 분포에 따른 범죄수
   4. 지역별 범죄종류(살인,절도,폭행 등)에 따른 발생 수
4. 서울 지역 행정구역 별 위도 및 경도 데이터와 folium 함수를 이용하여 행정구역별 범죄 합계 색깔 별로 표시

#### 추후계획

1. 시각화
2. 2014~2017년도 데이터를 통한 2018년 범죄 발생 수 예측을 위한 모델링

