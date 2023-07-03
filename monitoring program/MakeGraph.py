import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt


DATA_DIR = 'severstal-steel-defect-detection/' # data directory
TRAIN_IMG_DIR = DATA_DIR + 'train_images/'
TEST_IMG_DIR = DATA_DIR + 'test_images/'


# training set dataframe [ImageId, ClassId, EncodedPixels]
train_df = pd.read_csv(DATA_DIR + 'train.csv')


def main():
    # Separate dataframe by defect class
    defect_class1_df = train_df[train_df['ClassId'] == 1]
    defect_class2_df = train_df[train_df['ClassId'] == 2]
    defect_class3_df = train_df[train_df['ClassId'] == 3]
    defect_class4_df = train_df[train_df['ClassId'] == 4]

    class1_num = len(defect_class1_df)
    class2_num = len(defect_class2_df)
    class3_num = len(defect_class3_df)
    class4_num = len(defect_class4_df)

    num0 = 0
    num1 = 0
    num2 = 0
    num3 = 0

    counts = train_df['ImageId'].value_counts()
    for i in range(len(counts)):
        if counts[i] == 1:
            num1 += 1
        elif counts[i] == 2:
            num2 += 1
        else:
            num3 += 1

    num0 = 12568 - num1 - num2 - num3

    showNumOfDefects(class1_num, class2_num, class3_num, class4_num)
    # showNumOfMultiClass(num0, num1, num2, num3)

    
def showNumOfDefects(class1_num, class2_num, class3_num, class4_num):
    classes = ['Pitted(1)', 'Inclusion(2)', 'Scratches(3)', 'Patches(4)']
    values = [class1_num, class2_num, class3_num, class4_num]

    plt.title('Distribution of defect types')
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(values, labels=classes, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')       
    plt.savefig("graph.png")


def showNumOfMultiClass(num0, num1, num2, num3):
    nums = ['0', '1', '2', '3']
    values = [num0, num1, num2, num3]

    plt.title('Distribution of Number of defects in each image')
    plt.xlabel('Number of defects in each image')
    plt.ylabel('Number')

    plt.bar(nums, values)

    for i in range(len(nums)):
        plt.text(nums[i], values[i], values[i],
                fontsize = 9,
                color='black',
                horizontalalignment='center',
                verticalalignment='bottom')

    plt.show()

if __name__ == '__main__':
    main()