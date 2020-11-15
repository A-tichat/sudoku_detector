from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2
import os


def sudoku_detect(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.resize(gray, (480, 480))
    # dst = cv2.fastNlMeansDenoising(gray)
    # cv2.imshow('11', gray)
    # cv2.imshow('lo', dst)
    # cv2.waitKey(0)

    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 5)
    thresh = cv2.bitwise_not(thresh)
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    puzzleCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            puzzleCnt = approx
            break

    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
    return (puzzle, warped)


def convert_binary(gray_image):
    ''' Convert gray image to binary image'''
    image = cv2.fastNlMeansDenoising(gray_image)
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return image


def split_cell(blended_image):
    x_size = blended_image.shape[0] // 9
    y_size = blended_image.shape[1] // 9

    for x in range(0, 9):
        for y in range(0, 9):
            gray_cell = blended_image[x_size *
                                      x:x_size*(x+1), y_size*y:y_size*(y+1)]
            thresh = cv2.threshold(
                gray_cell, 0, 240, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            digit = clear_border(thresh)
            digit = cv2.erode(digit, np.ones((3, 3), np.uint8), iterations=1)
            yield cv2.dilate(digit, np.ones((1, 1), np.uint8), iterations=1)


def debug(image, color_map=None):
    if color_map is None:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, color_map)
    plt.xticks([]), plt.yticks([])
    plt.show()


def render(image, scale):
    image_list = image.flatten()
    fig = plt.figure(figsize=scale)
    for x in range(scale[0]):
        for y in range(scale[1]):
            index = x * scale[1]+y
            fig.add_subplot(scale[0], scale[1], index + 1),
            plt.imshow(cv2.cvtColor(image_list[index], cv2.COLOR_BGR2RGB))
            plt.xticks([]), plt.yticks([])
    plt.show()
