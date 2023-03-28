import os
import pandas as pd
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt

def defect_exec(address):
    input_dir = address #'severstal-steel-defect-detection/'
    print(os.listdir(input_dir))

    train_df = pd.read_csv(input_dir+'train.csv')
    sample_df = pd.read_csv(input_dir + 'sample_submission.csv')
    train_df.head()

    num_train = len(train_df)
    num_train

    class_by_1 = train_df[train_df['ClassId'] == 1]
    class_by_2 = train_df[train_df['ClassId'] == 2]
    class_by_3 = train_df[train_df['ClassId'] == 3]
    class_by_4 = train_df[train_df['ClassId'] == 4]
    print(len(class_by_1))
    class_by_1.head()
    print(len(class_by_2))
    class_by_2.head()
    print(len(class_by_3))
    class_by_3.head()
    print(len(class_by_4))
    class_by_4.head()

    TRAIN_IMGS_PATH = os.path.join(input_dir, 'train_images/')
    TEST_IMGS_PATH = os.path.join(input_dir, 'test_images/')
    print(TRAIN_IMGS_PATH)
    print(TEST_IMGS_PATH)
    train_fp = sorted(glob(TRAIN_IMGS_PATH + '*.jpg'))
    test_fp = sorted(glob(TEST_IMGS_PATH + '*.jpg'))
    train_0_name = train_fp[0].split('/')[-1]
    train_0_name
    for i in range(20):
        # 데이터 프레임의 경우
        # 파일 경로 어레이의 경우
        print(train_df.ImageId[i],'-------', train_fp[i])
    train_df[train_df['ImageId'] == '00031f466.jpg' ]
    print('length of DF : ', len(train_df.ImageId))
    print('length of img files : ', len(train_fp))
    df_grBy = train_df.groupby('ImageId')['EncodedPixels'].count()
    df_grBy
    none_defect = df_grBy[df_grBy==0].count()
    single_defect = df_grBy[df_grBy==1].count()
    double_defect = df_grBy[df_grBy==2].count()
    triple_defect = df_grBy[df_grBy==3].count()
    quatt_defect = df_grBy[df_grBy==4].count()
    print('the number of none defect imgs : ', none_defect)
    print('the number of single defect imgs : ', single_defect)
    print('the number of double defect imgs : ', double_defect)
    print('the number of triple defect imgs : ', triple_defect)
    print('the number of quattro defect imgs : ', quatt_defect)

    df_grBy[df_grBy==3]

    labels = 'none', 'single', 'double', 'triple', 'quattro'
    sizes = [none_defect, single_defect, double_defect, triple_defect, quatt_defect]

    fig, ax = plt.subplots(figsize=(6,6))
    ax.hist(df_grBy)
    ax.set_title('Number of Labels per img.')
    print(df_grBy.mean())
    print(df_grBy.max())
    defect1 = class_by_1.EncodedPixels.count()
    defect2 = class_by_2.EncodedPixels.count()
    defect3 = class_by_3.EncodedPixels.count()
    defect4 = class_by_4.EncodedPixels.count()
    print(defect1)
    print(defect2)
    print(defect3)
    print(defect4)

    labels = 'defect1', 'defect2', 'defect3', 'defect4'
    sizes = [defect1, defect2, defect3, defect4]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title('Defect Types')
    plt.show()

    enco1_str = train_df.EncodedPixels[0]
    enco1_str

    enco1_arr = enco1_str.split(' ')
    enco1_arr[:10]

    enco1_pos = map(int, enco1_arr[0::2])
    enco1_len = map(int, enco1_arr[1::2])
    enco1_pos

    temp_pos = list(enco1_pos)
    temp_len = list(enco1_len)
    temp_pos[:10]
    temp_len[:10]

    enco1_pos = map(int, enco1_arr[0::2])
    enco1_len = map(int, enco1_arr[1::2])
    for po, le in zip(enco1_pos, enco1_len):
        print(po, '<---->',le)
    img1 = cv2.imread(train_fp[0])
    fig, ax = plt.subplots(figsize=(15,5))
    ax.axis('off')
    ax.imshow(img1)
    plt.show()
    print(train_fp[0])
    TRAIN_IMGS_PATH + train_df.ImageId[0]
    print(train_fp[3])
    TRAIN_IMGS_PATH + train_df.ImageId[3]
    print(img1.shape)
    mask_1d = np.zeros(img1.shape[0] * img1.shape[1], dtype=np.uint8)
    enco1_pos = map(int, enco1_arr[0::2])
    enco1_len = map(int, enco1_arr[1::2])
    print(mask_1d.sum())
    for po, le in zip(enco1_pos, enco1_len):
        mask_1d[po:po+le-1]=255
    mask_1d.sum()
    mask2d = mask_1d.reshape(img1.shape[0], img1.shape[1], order='F')
    fig, ax = plt.subplots(2,1, figsize=(15,5))

    ax[0].axis('off')
    ax[0].set_title('original')
    ax[0].imshow(img1)

    ax[1].axis('off')
    ax[1].set_title('mask')
    ax[1].imshow(mask2d)
    mask_contour, hier_ = cv2.findContours(mask2d, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    print(len(mask_contour))
    len(mask_contour[0])

    for i in range(len(mask_contour)):
        img1_poly = cv2.polylines(img1, mask_contour[i], True, (255,100,0), 2, cv2.LINE_AA)
    fig, ax = plt.subplots(figsize=(15,5))

    ax.axis('off')
    ax.set_title('original')
    ax.imshow(img1_poly)

    def decode_PosLen(encodedStr):
        str_encodedArr = encodedStr.split(" ")
        
        int_posArr = map(int, str_encodedArr[0::2])
        int_lenArr = map(int, str_encodedArr[1::2])
        
        return int_posArr, int_lenArr

    def getcontourArr(imgShape, int_posArr, int_lenArr):
        # imgShape : tuple data
        # int_posArr, int_lenArr >> map 상태이다.
        int_contourArr = []
        
        # 1d img
        maskOneD = np.zeros(imgShape[0]*imgShape[1], dtype=np.uint8)
        for po, le in zip(int_posArr, int_lenArr):
            maskOneD[po:po+le-1] = 255
        
        # reshape 1d to 2d
        mask = maskOneD.reshape(imgShape[0], imgShape[1], order='F')
        
        # get contour pixels
        int_contourArr, hier_ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        return int_contourArr

    def show_mask(dfRow):
        
        imgID = train_df.ImageId[dfRow] 
        encodedStr = train_df.EncodedPixels[dfRow]
        classID = train_df.ClassId[dfRow]
        
        # read img.
        str_imgPath = TRAIN_IMGS_PATH + imgID
        img = cv2.imread(str_imgPath)
        if img is None:
            print("img is empty.")
            return -1
        
        # decode str_encodArr to int_decodeArr, position and length
        int_posArr, int_lenArr = decode_PosLen(encodedStr)
        
        # extract mask contour pixel data.
        int_contourArr = getcontourArr(img.shape ,int_posArr, int_lenArr) 

        # draw the pixels on origianl img.
        for i in range(len(int_contourArr)):
            img_poly = cv2.polylines(img, int_contourArr[i], True, (255,100,0), 2, cv2.LINE_AA)
        
        # plot img
        fig, ax = plt.subplots(figsize=(15,5))
        ax.axis('off')
        ax.set_title("CLASS ID : " + str(classID) +" -------------- IMG ID : "+ imgID)
        ax.imshow(img_poly)  
        return None

    dfRow = 2222
    show_mask(dfRow)
    df_grBy[df_grBy==3]
    train_df[train_df.ImageId == "db4867ee8.jpg"]

    for dfRow in range(6101, 6104):
    #     show_mask(train_df.ImageId[i], train_df.EncodedPixels[i])
        show_mask(dfRow)

    for dfRow in class_by_1.index[:3]:
        show_mask(dfRow)

    for dfRow in class_by_2.index[:3]:
        show_mask(dfRow)

    for dfRow in class_by_3.index[:3]:
        show_mask(dfRow)

    for dfRow in class_by_4.index[:3]:
        show_mask(dfRow)