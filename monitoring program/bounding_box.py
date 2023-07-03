import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


DATA_DIR = 'severstal-steel-defect-detection/' # 데이터 위치
TRAIN_IMG_DIR = DATA_DIR + 'train_images/' # 학습 이미지 파일의 위치
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)] # 결함에 따른 표시 색깔 설정


def main():
    # bounding_box.csv 파일 데이터프레임 읽기
    coor_df = pd.read_csv('bounding_box.csv')

    # bounding_box.csv 파일로부터 읽은 값들을 준비
    # boundingBoxData = (이미지 이름, 클래스 번호, 1번 결함 좌표, 2번 결함 좌표, 3번 결함 좌표, 4번 결함 좌표)
    boundingBoxData = prepareCoor(coor_df)

    # 이미지 이름과 위의 데이터를 변수로 받아 bounding box가 적용된 학습 이미지 생성
    imageId = '0025bde0c.jpg'
    img = createImgWithBoundingBox(imageId, boundingBoxData)


# bounding_box.csv 파일로부터 읽은 값들을 준비
def prepareCoor(coor_df):
    imageIds = []
    classIds = []
    coor1s = []
    coor2s = []
    coor3s = []
    coor4s = []
    coorList = []

    for imageId, classId, coor1, coor2, coor3, coor4 in tqdm(coor_df.values):
        imageIds.append(imageId)
        classId = classId.split(',')
        for i in range(len(classId)):
            classId[i] = int(classId[i])
        classIds.append(classId)
        coorList.append(coor1)
        coorList.append(coor2)
        coorList.append(coor3)
        coorList.append(coor4)
        for i in range(4):
            if coorList[i] != '0':
                coorStr = coorList[i]
                coorStr = coorStr.replace('[', '').replace(']', '').replace("'", '')
                coors = coorStr.split(', ')
                for j in range(len(coors)):
                    coors[j] = coors[j].split(' ')
                    for k in range(len(coors[j])):
                        coors[j][k] = int(coors[j][k])
            else:
                coors = 0
            exec('coor' + str(i+1) + 's.append(coors)')

        coorList.clear()

    return imageIds, classIds, coor1s, coor2s, coor3s, coor4s


# Bounding box가 적용된 학습 이미지 생성
# boundingBoxData = (이미지 이름, 클래스 번호, 1번 결함 좌표, 2번 결함 좌표, 3번 결함 좌표, 4번 결함 좌표)
def createImgWithBoundingBox(imageId, boundingBoxData):
    # 변수로 입력된 이미지 이름에 해당하는 학습 이미지 읽기
    img = cv2.imread(TRAIN_IMG_DIR + imageId)
    index = boundingBoxData[0].index(imageId)
    img_with_bounding_box = img[:]

    # 마스크 및 클래스 리스트 생성
    boundingBoxMask = []
    classIds = []

    # 한 이미지에 결함이 여러 개 있는 경우를 고려해서 마스크 생성
    for i in range(len(boundingBoxData[1][index])):
        mask, classId = createBoundingBoxMask(boundingBoxData, index, i)
        boundingBoxMask.append(mask)
        classIds.append(classId)

    # 마스크의 각 결함들에 대한 등치선을 추출하고 학습 이미지와 결합
    for i in range(len(boundingBoxMask)):
        contours, hierarchy = cv2.findContours(boundingBoxMask[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for j in range(len(contours)):
            # img_with_bounding_box = cv2.polylines(img_with_bounding_box, contours[j], True, COLORS[classIds[i]-1], 2, cv2.LINE_AA)
            img_with_bounding_box = cv2.polylines(img_with_bounding_box, contours[j], True, (0, 0, 255), 2, cv2.LINE_AA)
    
    # 테스트
    # fig, ax = plt.subplots(figsize=(15, 5))
    # ax.axis('off')
    # ax.imshow(img_with_bounding_box)
    # plt.show()

    return img_with_bounding_box


# Bounding box가 적용된 마스크 생성
def createBoundingBoxMask(boundingBoxData, index, i):
    mask = np.zeros((256, 1600), dtype=np.uint8)
    classId = boundingBoxData[1][index][i]
    coor = boundingBoxData[classId+1][index]

    # 인덱스 오류 방지
    for i in range(len(coor)):
        coor[i][2] = coor[i][2] - 1
        coor[i][3] = coor[i][3] - 1

    # 각 bounding box 좌표들을 마스크에 기입
    for i in range(len(coor)):
        mask[coor[i][1], coor[i][0]:coor[i][0]+coor[i][2]] = 255
        mask[coor[i][1]+coor[i][3], coor[i][0]:coor[i][0]+coor[i][2]] = 255
        mask[coor[i][1]:coor[i][1]+coor[i][3], coor[i][0]] = 255
        mask[coor[i][1]:coor[i][1]+coor[i][3], coor[i][0]+coor[i][2]] = 255

    return mask, classId


if __name__ == '__main__':
    main()