# SteelDefectDetection
2022 후기 졸업과제 - 졸업시켜조

# 연구 개요

## 연구 배경

- 고품질 철판에 대한 수요 상승
- 철판 제조 과정에서 결함은 필연적이나, 일부 결함 발생시 철판 품질에 치명적인 영향
- 결함 검사는 주로 수동으로 이루어져 비용이 크기 때문에 자동으로 식별하는 방법을 필요로함

## 연구 목표

- 철판 이미지를 전처리해 CNN 모델을 바탕으로 철판 결함 검출
- 결과를 시각화 하고자 함

# 연구 상세

- 사용한 데이터셋 및 이미지 전처리 방법
    

    - 데이터셋 : Kaggle - Severstal: Steel Defect Detection
        - 1600 x 256 사이즈의 이미지로 구성
        - 4가지 종류의 결함
        - 하나의 이미지에 여러개, 여러 종류의 결함으로 구성
    - 이미지 전처리
        - Resailing, Flip, Cutout 등의 Augmentation 기법을 이용
        - Augmentation을 이용해 각 클래스의 데이터의 balance를 맞춰 줌

- 활용 모델 및 Class 별 Confusion Matrix
    
    
    - Base 모델 : Xception
        - 선행 연구에서 Xception과 ResNet50을 활용한 연구가 우수한 성능 지표를 나타냄
        - 이중 Xception이 레이어 수정이 쉽고 간편해 선정
    
   
    - Confusion Matrix
        - 성능 지표는 f1-score
        - 기존 데이터셋의 4가지 결함과(Class 1 ~ 4) 결함 없음(Class 0)으로 총 5개의 클래스로 학습 진행
        - Xception을 Base Model로 튜닝해 학습한 모델에 적용한 f1-score는 좌측의 그림과 같음

# 결과 프로그램

- Pyqt5를 이용해 프로그램 설계
- 파일 불러오기를 통해 이미지를 전체적으로 불러옴
- 실행하기 버튼을 이용해 이미지의 결함 부분을 검출한 뒤 Bounding Box 형태로 나타냄
- 각 결함의 확대된 이미지를 새창에서 출력
