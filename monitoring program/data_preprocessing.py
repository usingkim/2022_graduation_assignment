import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers.experimental import preprocessing as pp
import random
import bounding_box


DATA_DIR = 'severstal-steel-defect-detection/' # csv 파일의 위치
TRAIN_IMG_DIR = DATA_DIR + 'train_images/' # 학습 이미지 파일의 위치
CROPPED_IMG_DIR = DATA_DIR + 'cropped_images/' # 결함 부분이 존재하는 크롭된 이미지 파일의 위치
N_CROPPED_IMG_DIR = DATA_DIR + 'n_cropped_images/' # 결함 부분이 존재하지 않는 크롭된 이미지 파일의 위치
AUGMENTED_IMG_DIR = DATA_DIR + 'augmented_images/' # Augmented된 이미지 파일의 위치
RESULT_DIR = DATA_DIR + 'result/'
TEST_IMG_DIR = 'test_images2/' # 테스트용


def main():
    train_df = pd.read_csv(DATA_DIR + 'train.csv') # train.csv 파일 데이터프레임

    # 클래스 별로 데이터프레임을 분리
    train_df_class1 = train_df[train_df['ClassId'] == 1]
    train_df_class2 = train_df[train_df['ClassId'] == 2]
    train_df_class3 = train_df[train_df['ClassId'] == 3]
    train_df_class4 = train_df[train_df['ClassId'] == 4]

    # 클래스 2의 데이터가 지나치게 적으므로 데이터를 복제해서 늘림
    # 데이터 증강을 통해 데이터 중복 문제를 보완
    sample2 = train_df_class2
    for _ in range(3):
        sample2 = pd.concat([sample2, train_df_class2], ignore_index=True)

    # 가장 적은 데이터를 가진 데이터프레임 4를 샘플링 기준으로 삼음
    min_class_num = len(train_df_class4)
    sample4 = train_df_class4

    # 기준으로 삼은 수 만큼 다른 데이터프레임들을 샘플링
    sample1 = train_df_class1.sample(n=min_class_num)
    sample2 = sample2.sample(n=min_class_num)
    sample3 = train_df_class3.sample(n=min_class_num)

    # 샘플링 된 데이터프레임들을 합쳐서 하나의 데이터프레임 생성
    # train_df_sample = pd.concat([sample1, sample2, sample3, sample4], ignore_index=True)

    # 데이터프레임을 읽고 ImageId, ClassId, EncodedPixels, augmentation 된 이미지의 nparray 반환
    # imageIds, classIds, encodedPixelsList, imgs, n_imgs, img_nums = prepareData(train_df_sample)

    # 결함 부분이 존재하는/존재하지 않는 이미지 데이터 augmentation
    # augmented_imgs = augmentation(imgs) # 결함 존재 (Class 1-4)
    # n_augmented_imgs = augmentation(n_imgs) # 결함 없음 (Class 0)

    # augmentation을 수행한 이미지 저장
    # saveAugmentedImgs(imageIds, classIds, augmented_imgs, img_nums)

    # 이미지 저장(결과 출력용)
    coor_df = pd.read_csv('bounding_box.csv')
    boundingBoxData = bounding_box.prepareCoor(coor_df)
    imageIds, classIds, encodedPixelsList, imgs, n_imgs, img_nums = prepareData(train_df)
    saveResult(imageIds, classIds, imgs, img_nums, boundingBoxData)

    # 테스트
    # test(train_df_sample) 


# 테스트
def test(train_df):
    imageIds, classIds, encodedPixelsList, imgs, n_imgs, img_nums = prepareData(train_df)

    imgs = np.array(imgs)
    augmented_imgs = augmentation(imgs)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        ax = plt.subplot(10, 1, i+1)
        plt.subplots_adjust(hspace=1)
        plt.imshow(imgs[i])
    plt.show()

    plt.figure(figsize=(10, 10))
    for i in range(10):
        ax = plt.subplot(10, 1, i+1)
        plt.subplots_adjust(hspace=1)
        plt.imshow(augmented_imgs[i])
    plt.show()

    # saveAugmentedImgs(imageIds, classIds, augmented_imgs)

    return None


# 데이터 준비
def prepareData(train_df):
    # 이미지 이름, 결함 클래스, 인코딩 픽셀, 결함 있는 이미지, 결함 없는 이미지, 이미지 넘버
    imageIds = []
    classIds = []
    encodedPixelsList = []
    imgs = []
    n_imgs = []
    img_nums = []

    # 이미지를 전부 불러오기 위한 리스트
    imgPathList = []

    # csv 데이터 읽기
    for imageId, classId, encodedPixels in tqdm(train_df.values):
        # 결함 크롭 이미지를 전부 불러오기
        imgPathList = selectAllImg(imageId, classId)
        if len(imgPathList) > 0:
            for i in range(len(imgPathList)):
                imageIds.append(imageId)
                classIds.append(classId)
                encodedPixelsList.append(encodedPixels)
                imgPath = imgPathList[i]
                img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
                img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
                imgs.append(img)
                img_nums.append(imgPath[-5])
        # 결함이 없는 이미지 전부 불러오기
        imgPathList.clear()
        imgPathList = selectAllImg(imageId, 'n')
        if len(imgPathList) > 0:
            for i in range(len(imgPathList)):
                n_imgPath = imgPathList[i]
                n_img = cv2.imread(n_imgPath, cv2.IMREAD_COLOR)
                n_img = cv2.resize(n_img, dsize=(128, 128), interpolation=cv2.INTER_AREA)
                n_imgs.append(n_img)

    imageIds = np.array(imageIds)
    classIds = np.array(classIds)
    encodedPixelsList = np.array(encodedPixelsList)
    imgs = np.array(imgs)
    n_imgs = np.array(n_imgs)

    return imageIds, classIds, encodedPixelsList, imgs, n_imgs, img_nums


# 이미지 데이터에 augmentation 수행
def augmentation(imgs):
    # 이미지 데이터에 적용시킬 augmentation 기법 설정
    generator = tf.keras.Sequential([
        pp.RandomFlip('horizontal'),
        pp.RandomFlip('vertical'),
        pp.RandomContrast(0.1),
        pp.RandomTranslation(0.05, 0.05, fill_mode='reflect'),
        pp.RandomRotation(0.5, fill_mode='reflect'),
        pp.Rescaling(1./255)
    ])

    flag = 0
    for it in tqdm(batch(imgs)):
        if flag != 0:
            # 이미지 데이터에 cutout 수행
            it = tfa.image.random_cutout(generator(it), (64, 64), 0)   
            augmented_imgs = np.vstack((augmented_imgs, it))
        else:
            augmented_imgs = tfa.image.random_cutout(generator(it), (64, 64), 0)
            flag = 1     

    return augmented_imgs


# 한 번에 batch_size 만큼의 이미지의 augmentation을 수행
def batch(iterable, batch_size=128):
  for ndx in range(0, len(iterable), batch_size):
    yield iterable[ndx:ndx + batch_size]


# 한 결함에 여러 이미지 데이터가 존재하는 경우 무작위 선택
def selectRandomImg(imageId, classId):
    flag = 0
    stuckLimit = 0

    if classId == 'n':
        dir = N_CROPPED_IMG_DIR
    else:
        dir = CROPPED_IMG_DIR

    imgPath = dir + imageId[:-4] + '_' + str(classId) + '_' + str(random.randint(0, 10)) + '.png'
    while flag == 0 and stuckLimit <= 10:
        if os.path.isfile(imgPath):
            flag = 1
        elif classId != 'n':
            imgPath = dir + imageId[:-4] + '_' + str(classId) + '_' + str(random.randint(0, 10)) + '.png'
        else:
            imgPath = dir + imageId[:-4] + '_' + str(classId) + '_' + str(random.randint(0, 10)) + '.png'
            stuckLimit += 1

    return imgPath


# 한 이미지에 대하여 모든 결함 크롭 이미지 가져오기
def selectAllImg(imageId, classId):
    imgPathList = []

    if classId == 'n':
        dir = N_CROPPED_IMG_DIR
    else:
        dir = CROPPED_IMG_DIR

    for i in range(20):
        imgPath = imgPath = dir + imageId[:-4] + '_' + str(classId) + '_' + str(i) + '.png'
        if os.path.isfile(imgPath):
            imgPathList.append(imgPath)

    return imgPathList


# Augmentation이 적용된 이미지 저장
def saveAugmentedImgs(imageIds, classIds, augmented_imgs, img_nums):
    if not os.path.exists(AUGMENTED_IMG_DIR):
        os.mkdir(AUGMENTED_IMG_DIR)

    output_imgs = np.array(255 * augmented_imgs, dtype='uint8')

    for i in range(len(output_imgs)):
        name = imageIds[i][:-4] + '_' + str(classIds[i]) + '_' + img_nums[i] +'.png'
        cv2.imwrite(AUGMENTED_IMG_DIR + name, output_imgs[i])


# Bounding box 적용된 이미지 및 전처리된 이미지 저장
def saveResult(imageIds, classIds, imgs, img_nums, boundingBoxData):
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    dup_list = []

    # # Bounding box 적용된 이미지 저장
    # for imageId in tqdm(imageIds):
    #     if imageId not in dup_list:
    #         name = imageId
    #         img_with_bounding_box = bounding_box.createImgWithBoundingBox(imageId, boundingBoxData)
    #         cv2.imwrite(RESULT_DIR + name, img_with_bounding_box)
        
    #     dup_list.append(imageId)

    # 전처리된 이미지 저장
    for i in tqdm(range(len(imageIds))):
        if not os.path.exists(RESULT_DIR + imageIds[i][:-4]):
            os.mkdir(RESULT_DIR + imageIds[i][:-4])

        name = imageIds[i][:-4] + '_' + str(classIds[i]) + '_' + img_nums[i] + '.png'
        cv2.imwrite(RESULT_DIR + imageIds[i][:-4] + '/' + name, imgs[i])
        

if __name__ == '__main__':
    main()