# -*- coding: UTF-8 -*-
import glob
import cv2
import os
import shutil
import numpy as np


class KNN(object):
    def __init__(self):
        self.k = 0
        self.labels = []
        self.samples = []

    @staticmethod
    def make_dir(base_path, dirs):
        """ 创建存储单个字符的目录 """
        if type(dirs) == str:
            dirs = list(dirs)
        for s in dirs:
            _path = os.path.join(base_path, s)
            if not os.path.isdir(_path):
                os.makedirs(_path)

    def split_letter(self, img_path, letters):
        """ 读入图片名称列表及图片内字符列表, 将单个字符拆分开并存入相应目录下
        :param img_path: 图片名称列表
        :param letters: 图片内字符列表
        :return: none
        """
        self.make_dir("number", letters)
        for path in img_path:
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (1, 1), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 33, 2)

            image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contours_lst = []
            for cnt in contours:
                [x, y, w, h] = cv2.boundingRect(cnt)
                if h > 10:
                    contours_lst.append([x, y, w, h])

            # 按照 x 坐标排序
            contours_sorted = sorted(contours_lst, key=lambda t: t[0])

            for i in range(len(letters)):
                [x, y, w, h] = contours_sorted[i]
                thresh1 = thresh[y:y + h, x:x + w]
                thresh1 = cv2.resize(thresh1, (20, 40))

                # 每一个数字存入对应数字的文件夹
                number_path1 = "number/%s/%d" % (letters[i], self.k) + '.jpg'
                self.k += 1

                # 归一化
                normalized_roi1 = thresh1 / 255.

                # 把图片展开成一行，然后保存到samples, 图片信息及标签需要成对保存，类似键值对
                sample1 = normalized_roi1.reshape((1, 800))
                self.samples.append(sample1[0])
                self.labels.append(float(ord(letters[i])))

                cv2.imwrite(number_path1, thresh1)

    def save_letter(self):
        """ 读取预先准备的存储英文|数字字符的图片，保存至 .npy 格式的二进制文件 """
        # 大写字母
        img_path = glob.glob("numbers/upper_case_letter_*")
        upper_letter = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.split_letter(img_path, upper_letter)

        # 小写字母
        img_path = glob.glob("numbers/lower_case_letter_*")
        upper_letter = list("abcdefghijklmnopqrstuvwxyz")
        self.split_letter(img_path, upper_letter)

        # 阿拉伯数字
        img_path = glob.glob("numbers/number_*")
        num = list("0123456789")
        self.split_letter(img_path, num)

        samples = np.array(self.samples, np.float32)
        labels = np.array(self.labels, np.float32)
        labels = labels.reshape((labels.size, 1))

        np.save('samples.npy', samples)
        np.save('label.npy', labels)

    @staticmethod
    def get_binary(img, method):
        """ 灰度图转二值化图
        :param img:  灰度图
        :param method: 方法
        :return: 二值图
        """
        if method == 0:
            # 自适应二值化
            thresh = cv2.adaptiveThreshold(img, 255, 1, 1, 33, 2)

            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
            dilated = cv2.dilate(thresh, kernel)
            return dilated
        else:
            # 简单二值化
            _, dilated = cv2.threshold(img, 155, 255, cv2.THRESH_BINARY_INV)
            return dilated

    def letter_recognition(self):
        """ 使用 knn 算法进行英文字符识别 """
        samples = np.load('samples.npy')
        labels = np.load('label.npy')

        # 将存储的样本全部用来训练模型
        train_label = labels
        train_input = samples

        model = cv2.ml.KNearest_create()
        model.train(train_input, cv2.ml.ROW_SAMPLE, train_label)

        img = cv2.imread('./images/003.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dilated = self.get_binary(gray, 1)

        # 轮廓提取
        image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imwrite('./images/003_thresh.jpg', dilated)

        # phrase = {}
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            thresh1 = dilated[y:y + h, x:x + w]
            thresh1 = cv2.resize(thresh1, (20, 40))

            # 归一化像素值
            normalized_roi = thresh1 / 255.

            # 展开成一行让knn识别
            sample1 = normalized_roi.reshape((1, 800))
            sample1 = np.array(sample1, np.float32)

            # knn识别
            retval, results, neigh_resp, dists = model.findNearest(sample1, 1)
            number = results.ravel()

            # 若被识别的区域与模板之间的距离小于预设阈值
            if dists[0] < 145:
                print(str(chr(number)), x, y, w, h)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(img, str(chr(number)), (x, y), 1, 1.4, (0, 0, 255), 1, cv2.LINE_AA)

        # cv2.imwrite('./images/003_result.png', img)


if __name__ == '__main__':
    if os.path.isdir("number"):
        shutil.rmtree("number")
    knn = KNN()
    knn.save_letter()
    knn.letter_recognition()