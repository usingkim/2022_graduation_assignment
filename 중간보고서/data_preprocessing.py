from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.layers.experimental import preprocessing as pp


DATA_DIR = 'severstal-steel-defect-detection/' # csv 파일의 위치
TRAIN_IMG_DIR = DATA_DIR + 'train_images/' # 학습 이미지 파일의 위치


def main():
    train_df = pd.read_csv(DATA_DIR + 'train.csv') # train.csv 파일 데이터프레임

    # 데이터프레임을 읽고 ClassId, EncodedPixels, 전처리 및 augmentation 된 이미지의 array 반환
    classIds, encodedPixelsList, augmented_imgs = preprocess(train_df)

    # 이미지 전처리 테스트
    # augmented_imgs = test(train_df) 


# 이미지 전처리 테스트
def test(train_df):
    imgs = []

    for i in range(5):
        img = cv2.imread(TRAIN_IMG_DIR + train_df.ImageId[i], cv2.IMREAD_COLOR)
        imgs.append(img)

    imgs = np.array(imgs)
    augmented_imgs = augmentation(imgs)

    plt.figure(figsize=(10, 10))
    for i in range(5):
        ax = plt.subplot(1, 5, i+1)
        plt.imshow(augmented_imgs[i])
    plt.show()

    return None


# 데이터 전처리 통합
def preprocess(train_df):
    imgs = []
    classIds = []
    encodedPixelsList = []

    # csv 데이터 읽기
    for imageId, classId, encodedPixels in tqdm(train_df.values):
        classIds.append(classId)
        encodedPixelsList.append(encodedPixels)
        img = cv2.imread(TRAIN_IMG_DIR + "{}".format(imageId), cv2.IMREAD_COLOR)
        imgs.append(img)

    classIds = np.array(classIds)
    encodedPixelsList = np.array(encodedPixelsList)
    imgs = np.array(imgs)
    augmented_imgs = augmentation(imgs) # 이미지 데이터 전처리 및 augmentation

    return classIds, encodedPixelsList, augmented_imgs


# 이미지 데이터에 전처리 및 augmentation 수행
def augmentation(imgs):
    # 이미지 데이터 전처리 + 적용시킬 augmentation 기법 설정
    generator = tf.keras.Sequential([
        pp.Resizing(255, 255),
        pp.Rescaling(1./255),
        pp.RandomZoom(0.1, fill_mode='constant', fill_value=0),
        pp.RandomFlip('horizontal'),
        pp.RandomFlip('vertical')
    ])

    flag = 0
    for it in tqdm(batch(imgs)):
        if flag != 0:
            it = generator(it)
            augmented_imgs = np.vstack((augmented_imgs, it))      
        else:
            augmented_imgs = generator(it)
            flag = 1     
            
    return augmented_imgs


# 한 번에 batch_size 만큼의 이미지의 전처리 및 augmentation을 수행
def batch(iterable, batch_size=128):
  for ndx in range(0, len(iterable), batch_size):
    yield iterable[ndx:ndx + batch_size]


if __name__ == '__main__':
    main()
