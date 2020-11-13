from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2


class Sudoku:
    # Sudoku.image_path = path to image file
    # Sudoku.full_image = original image
    # Sudoku.image_gray = bird eye view gray image
    # Sudoku.image_binary = thresholded and denoised image
    # Sudoku.cells = array of all grid
    def __init__(self, path):
        self.image_path = path
        self.full_image = cv2.imread(self.image_path)
        self.sudoku_detect()
        self.convert_binary()
        self.split_cells()

    def sudoku_detect(self):
        gray = cv2.cvtColor(self.full_image, cv2.COLOR_BGR2GRAY)

        # Thrashold
        blurred = cv2.GaussianBlur(gray, (7, 7), 3)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 5)
        thresh = cv2.bitwise_not(thresh)

        cnts = self.find_contour(thresh)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        puzzleCnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                puzzleCnt = approx
                break

        # self.image_bgr = four_point_transform(
        #     self.full_image, puzzleCnt.reshape(4, 2))
        self.image_gray = four_point_transform(gray, puzzleCnt.reshape(4, 2))

    def convert_binary(self):
        ''' Convert gray image to binary image'''
        image = cv2.fastNlMeansDenoising(self.image_gray)
        image = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        self.image_binary = image

    def filter_number(self, cell):
        _, thresh = cv2.threshold(cell, 0, 255,
                                  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        thresh = clear_border(thresh)

        cnts = self.find_contour(thresh)
        mask = np.zeros(thresh.shape, np.uint8)

        # remove all
        if len(cnts) == 0:
            return self.remove_image(thresh)

        c = max(cnts, key=cv2.contourArea)

        if cv2.contourArea(c) < 20:
            return self.remove_image(thresh)

        cv2.drawContours(mask, [c], -1, 255, -1)
        return cv2.bitwise_and(thresh, thresh, mask=mask)

    def find_contour(self, thresh):
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        return imutils.grab_contours(cnts)

    def remove_image(self, thresh):
        return cv2.bitwise_and(thresh, np.zeros(thresh.shape, np.uint8))

    def split_cells(self):
        x_size = self.image_binary.shape[0] // 9
        y_size = self.image_binary.shape[1] // 9

        array_cell = np.zeros(shape=(9, 9), dtype=object)

        for x in range(0, 9):
            for y in range(0, 9):
                cell = self.image_binary[x_size *
                                         x:x_size * (x+1), y_size * y:y_size * (y+1)]
                array_cell[x][y] = self.filter_number(cell)
        self.cells = array_cell
