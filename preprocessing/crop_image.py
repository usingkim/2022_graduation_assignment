import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random


DATA_DIR = 'severstal-steel-defect-detection/' # csv 파일의 위치
TRAIN_IMG_DIR = DATA_DIR + 'train_images/' # 학습 이미지 파일의 위치
CROPPED_IMG_DIR = 'cropped_images/' # 크롭된 이미지 파일의 위치
N_CROPPED_IMG_DIR = 'n_cropped_images/' # 크롭된 결함 부분이 없는 이미지 파일의 위치
TEST_IMG_DIR = 'test_images/' # 테스트용
BOUNDING_BOX_MASK_DIR = 'boundingbox_images/' # Bounding box가 적용된 학습 이미지 파일의 위치

# 결함이 없는 이미지 생성용
SHAPE = (256, 1600) # 원본 이미지 shape (높이 * 너비)
TARGET_SHAPE = 256 # 크롭 이미지의 shape
CROPS = int(SHAPE[1]/TARGET_SHAPE) # 하나의 이미지에서 생성될 크롭될 이미지의 개수
OFFSET_MARGIN_WIDTH = int((SHAPE[1] - TARGET_SHAPE * CROPS)/2) # 오프셋 너비 Margin
OFFSET_MARGIN_HEIGHT = (int((SHAPE[0]-TARGET_SHAPE)/2), -int((SHAPE[0]-TARGET_SHAPE)/2)) # 오프셋 높이 Margin (256인 경우 전부)
OFFSETS = [OFFSET_MARGIN_WIDTH + TARGET_SHAPE * i for i in range(CROPS)] # 원본 이미지에서 크롭할 위치에 대한 오프셋


def main():
    train_df = pd.read_csv(DATA_DIR + 'train.csv') # train.csv 파일 데이터프레임
    coor_df = pd.read_csv('bounding_box.csv') # bounding_box.csv 파일 데이터프레임

    cropImages(train_df) # 이미지 크롭을 수행하고 결과를 출력

    # bounding_box.csv 파일 데이터 준비
    # boundingBoxData(imageIds, classIds, coor1s, coor2s, coor3s, coor4s)
    # boundingBoxData = prepareCoor(coor_df[:10])

    # Bounding box가 적용된 학습 이미지 생성
    # for i in range(10):
    #     createImgWithBoundingBox(train_df.ImageId[i], boundingBoxData)

    # test(train_df)


# 테스트
def test(train_df):
    test_df = pd.DataFrame(columns=['ImageId','class1Coor','class2Coor','class3Coor','class4Coor'])
    test_df.set_index('ImageId', inplace=True)

    coor_row = []
    df_row = []
    coor1s = []
    coor2s = []
    coor3s = []
    coor4s = []

    r = random.randint(0, 10)

    mask2D = encodedPixelsTo2DMask(train_df.EncodedPixels[r])
    coors = defineBoundingBoxes(mask2D)
    image = cv2.imread(TRAIN_IMG_DIR + train_df.ImageId[r], cv2.IMREAD_COLOR)
    imageId = train_df.ImageId[r]
    classId = train_df.ClassId[r]

    if not os.path.exists(TEST_IMG_DIR):
        os.mkdir(TEST_IMG_DIR)

    cv2.imwrite(TEST_IMG_DIR + imageId[:-4] + '_' + 'mask.png', mask2D)

    for i in range(len(coors)):
        name = imageId[:-4] + '_' + str(classId) + '_' + str(i) + '.png'
        cropped_img = image[coors[i][1]:coors[i][1] + coors[i][3], coors[i][0]:coors[i][0] + coors[i][2]]
        cv2.imwrite(TEST_IMG_DIR + name, cropped_img)
        coor_row.append(str(coors[i][0]) + ' ' + str(coors[i][1]) + ' ' + str(coors[i][2]) + ' ' + str(coors[i][3]))

    # 좌표 저장
    for i in range(1, 5):
        if i == classId:
            exec('coor' + str(classId) + 's' + '.append(coor_row)')
        else:
            exec('coor' + str(i) + 's' + '.append(0)')

    df_row = [imageId, coor1s[0], coor2s[0], coor3s[0], coor4s[0]]

    test_df.append(df_row, ignore_index=True)
    test_df.to_csv('test.csv')


# encodedPixels를 2D mask로 변환
def encodedPixelsTo2DMask(encodedPixels, shape=(256, 1600)):
    encodedList = encodedPixels.split(" ") # encodedPixels 문자열을 리스트로 변환
    decodedPos = map(int, encodedList[0::2])
    decodedLen = map(int, encodedList[1::2])

    mask1D = np.zeros(shape[0]*shape[1], dtype=np.uint8) # 1차원 마스크

    # encodedPixels 값을 디코딩
    for po, le in zip(decodedPos, decodedLen):
        mask1D[po:po+le-1] = 255

    mask2D = mask1D.reshape(shape[0], shape[1], order='F') # 1차원 마스크를 2차원 마스크로 변환

    return mask2D


# 각 결함에 대한 bounding box 좌표를 생성하고, 이를 통해 이미지의 결함 부분만을 추출하는데 사용함
def defineBoundingBoxes(mask2D):
    coors = []

    contours, hierarchy = cv2.findContours(mask2D, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # 마스크에 각 결함들에 대한 등치선을 추출
    contour_areas = [(cv2.contourArea(contour), contour) for contour in contours] # 마스크에 각 결함들에 대한 등치선 범위 추출
    contour_sizes = [contourArea[1] for contourArea in contour_areas] # bounding box 좌표 생성에 필요한 크기 값 추출

    # bounding box 좌표 생성에 필요한 좌표 습득
    for i in range(len(contour_sizes)):
        x, y, w, h = cv2.boundingRect(contour_sizes[i])

        coors.append((x, y, w, h))

    return coors


# 이미지 크롭을 수행
def cropImages(train_df):
    # 크롭된 이미지가 저장될 디렉토리 생성
    if not os.path.exists(CROPPED_IMG_DIR):
        os.mkdir(CROPPED_IMG_DIR)
    if not os.path.exists(N_CROPPED_IMG_DIR):
        os.mkdir(N_CROPPED_IMG_DIR)

    # bounding box 좌표 csv 파일 생성에 사용될 리스트
    imageIds = []
    classIds = []
    coor1s = []
    coor2s = []
    coor3s = []
    coor4s = []
    row = []
    
    # 이미지 중복 체크
    check_duplication = train_df.duplicated(['ImageId'], False)
    num = 0

    # csv 데이터 읽기
    for imageId, classId, encodedPixels in tqdm(train_df.values):
        img = cv2.imread(TRAIN_IMG_DIR + "{}".format(imageId))
        mask2D = encodedPixelsTo2DMask(encodedPixels)
        coors = defineBoundingBoxes(mask2D)

        # 좌표 값들을 기준으로 이미지에서 각 결함 부분만을 크롭하여 저장
        # 크롭된 이미지 파일명 = 이미지 이름_결함 클래스_순번.png
        # 마스크 이미지 파일명 = 이미지 이름_mask.png
        # bounding box 좌표를 리스트에 삽입
        cv2.imwrite(CROPPED_IMG_DIR + imageId[:-4] + '_' + 'mask.png', mask2D)
        for i in range(len(coors)):
            name = imageId[:-4] + '_' + str(classId) + '_' + str(i) + '.png'
            cropped_img = img[coors[i][1]:coors[i][1] + coors[i][3], coors[i][0]:coors[i][0] + coors[i][2]]
            cv2.imwrite(CROPPED_IMG_DIR + name, cropped_img)
            row.append(str(coors[i][0]) + ' ' + str(coors[i][1]) + ' ' + str(coors[i][2]) + ' ' + str(coors[i][3]))

        # 결함이 존재하지 않는 부분을 크롭하여 저장
        # 결함이 존재하지 않는 크롭된 이미지 파일명 = 이미지 이름_n_순번.png
        # 이미지가 중복되는 경우 제외하고 크롭 수행
        if check_duplication[num] == False:
            count = 0
            for i in range(CROPS):
                offset_img = img[:, OFFSETS[i]:OFFSETS[i]+TARGET_SHAPE]
                offset_mask = mask2D[:, OFFSETS[i]:OFFSETS[i]+TARGET_SHAPE]
                if offset_mask.max() == 0 and offset_img.max() >= 10:
                    n_name = imageId[:-4] + '_n_' + str(count) + '.png'
                    cv2.imwrite(N_CROPPED_IMG_DIR + n_name, offset_img)
                    count += 1

        # 이미지에 결함이 여러 개인 경우를 체크하며 좌표 저장
        if imageId in imageIds:
            exec('coor' + str(classId) + 's[imageIds.index(imageId)] = row[:]')
            exec('classIds[imageIds.index(imageId)] += ", " + str(classId)')
        else:
            for i in range(1, 5):
                if i == classId:
                    exec('coor' + str(i) + 's' + '.append(row[:])')
                else:
                    exec('coor' + str(i) + 's' + '.append(0)')

            imageIds.append(imageId)
            classIds.append(str(classId))

        num += 1
        row.clear()

    saveBoundingBoxes(imageIds, classIds, coor1s, coor2s, coor3s, coor4s)


# Bounding box 좌표 저장
def saveBoundingBoxes(imageIds, classIds, coor1s, coor2s, coor3s, coor4s):
    # bounding box 좌표 csv 파일 생성
    # csv 파일 형식 '이미지 이름, Class1 결함 좌표, Class2 결함 좌표, Class3 결함 좌표, Class4 결함 좌표'
    # 좌표 형식 'x좌표, y좌표, bounding box 너비, bounding box 높이'
    # 하나의 Class에 대해서 여러 개의 결함이 존재하는 경우, 좌표들은 ','로 구분
    df = pd.DataFrame({'ImageId':imageIds,
                    'classId':classIds,
                    'class1Coor':coor1s,
                    'class2Coor':coor2s,
                    'class3Coor':coor3s,
                    'class4Coor':coor4s})
    boundingBox_df = df.set_index('ImageId')
    boundingBox_df.to_csv('bounding_box.csv')


# bounding_box.csv 파일로부터 읽은 값들을 정리
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
# boundingBoxData(imageIds, classIds, coor1s, coor2s, coor3s, coor4s)
def createImgWithBoundingBox(imageId, boundingBoxData):
    if not os.path.exists(BOUNDING_BOX_MASK_DIR):
        os.mkdir(BOUNDING_BOX_MASK_DIR)

    img = cv2.imread(TRAIN_IMG_DIR + imageId)
    index = boundingBoxData[0].index(imageId)

    boundingBoxMask = np.zeros((256, 1600), dtype=np.uint8)

    # 한 이미지에 결함이 여러 개 있는 경우를 고려해서 마스크 생성
    for i in range(len(boundingBoxData[1][index])):
        mask = createBoundingBoxMask(boundingBoxData, index, i)
        boundingBoxMask += mask

    # 마스크의 각 결함들에 대한 등치선을 추출
    contours, hierarchy = cv2.findContours(boundingBoxMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 학습 이미지와 bounding box 마스크를 결합
    for i in range(len(contours)):
        img_with_bounding_box = cv2.polylines(img, contours[i], True, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imwrite(BOUNDING_BOX_MASK_DIR + imageId, img_with_bounding_box)


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

    return mask


if __name__ == '__main__':
    main()