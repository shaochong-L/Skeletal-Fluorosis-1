import cv2 as cv
import numpy as np
import math
import cv2
import imutils
from PIL import Image
import os
from matplotlib import pyplot as plt

img_path = "preprocessing/"

def pre_processing2(name):
    a = cv.imread(img_path + name, 0)
    a = cv.GaussianBlur(a, (5, 5), 0)

    ret, dst = cv.threshold(a, 100, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)
    print(dst)

    dst = cv.medianBlur(dst, 75)
    s = cv.getStructuringElement(cv.MORPH_CROSS, (10, 10))
    dst = cv.erode(dst, s, borderType=cv.BORDER_CONSTANT, borderValue=0)

    rows, cols = dst.shape[:2]
    i, c, h = cv.findContours(dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    dsts = [dst, dst, dst]
    dst1 = cv.merge(dsts)

    ch = c[-1]
    [vx, vy, x, y] = cv.fitLine(ch, cv.DIST_L2, 0, 1, 0.01)

    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    theta = math.atan(vy / vx) / math.pi * 180
    print(name,theta)

    image = cv2.imread(img_path + name)
    rotated = imutils.rotate(image, theta)
    cv2.imwrite(img_path + name, rotated)

    im = Image.open(img_path + name)
    width = im.width
    height = im.height
    middle_width = int(width/2)
    middle_height = int(height/2)

    x = middle_width - 375
    y = middle_height - 250
    w = 750
    h = 500
    region = im.crop((x, y, x + w, y + h))
    region.save(img_path + name)

for filename in os.listdir(img_path):
    pre_processing2(filename)
