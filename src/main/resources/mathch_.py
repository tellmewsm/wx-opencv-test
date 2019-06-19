#!/usr/bin/python
# -*- coding=utf-8 -*-
import cv2
import numpy as np
import sys


def find_contours(img):
    """ 读入图像数据，返回图像轮廓 """
    img = cv2.blur(img, (3, 3))
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img, 40, 80)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def match_img(basic_img, template_img, value=0.9):
    """ 模板匹配
    :param basic_img: 待匹配的图像
    :param template_img: 模板图像
    :param value: 图像匹配度阈值
    :return: rect
    """
    w, h = template_img.shape[:2]
    res = cv2.matchTemplate(basic_img, template_img, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(res)
    print(min_val, max_val, min_indx, max_indx)

    loc = np.where(res >= value)
    # print loc
    for pt in zip(*loc[::-1]):
        cv2.rectangle(basic_img, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
    cv2.imshow('Detected', basic_img)
    # cv2.imwrite('E:\\logo\\re.jpg', basic_img)
    # print("=========")
    return [pt[0], pt[1], w, h]


def get_ratio(contours1, contours2, thre=0.001):
    """ 待匹配的图像和模板的缩放比例
    :param contours1: 待匹配图像内轮廓
    :param contours2: 模板图像轮廓
    :param thre: 形状匹配度阈值
    :return: float
    """
    cnt2 = contours2[0]
    # print cnt2
    _, m_radius = cv2.minEnclosingCircle(cnt2)
    m_radius = int(m_radius)

    radio = 1
    for cnt1 in contours1:
        ret = cv2.matchShapes(cnt1, cnt2, 1, 0.0)

        if ret < thre:
            (x, y), radius = cv2.minEnclosingCircle(cnt1)
            center = (int(x), int(y))
            radius = int(radius)

            if abs(x - half_width) / half_width < 0.1:
                print(radius, m_radius)
                radio = float(radius) / m_radius
    return radio


img1 = cv2.imread(sys.path[0] + '/3.jpg')
height, width = img1.shape[:2]
print height, width
half_width = width // 2

img2 = cv2.imread(sys.path[0] + '/M0.jpg')

contours2 = find_contours(img2)

if len(contours2) != 1:
    print("选取的模板不符合规格！")

contours1 = find_contours(img1)

ratio = get_ratio(contours1, contours2)

x, y = img2.shape[:2]
img2 = cv2.resize(img2, (int(y * ratio), int(x * ratio)))
match_img(img1, img2)
# cv2.waitKey(0)

# ===
img2 = cv2.imread(sys.path[0] + '/M1.jpg')
x, y = img2.shape[:2]
img2 = cv2.resize(img2, (int(y * ratio), int(x * ratio)))

match_img(img1, img2)
# cv2.waitKey(0)

# ===
img2 = cv2.imread(sys.path[0] + '/M3.jpg')
x, y = img2.shape[:2]
img2 = cv2.resize(img2, (int(y * ratio), int(x * ratio)))

match_img(img1, img2)
# cv2.waitKey(0)
