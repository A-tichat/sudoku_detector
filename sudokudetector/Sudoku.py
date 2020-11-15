from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
# from image import debug
import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2


class Sudoku:

    def __init__(self, path):
        self.image_path = path
        self.sudoku_detect()
        self.image_binary = self.convert_binary(self.image_gray)
        self.cells = self.split_cells(self.image_binary)
        self.help = """Sudoku('path/to/file')\n
    Data\n
    \t.image_path = path to image file\n
    \t.image_gray = bird eye view gray image\n
    \t.image_binary = thresholded and denoised image\n
    \t.cells = array of all grid\n
    \t.help = detail of this object\n
    Function\n
    \t.load_image() = load original image from self.image_path\n
    \t\treturn image_object\n
    \t.sudoku_detect() = transfrom gray image to bird eye view\n
    \t\treturn None\n
    \t.convert_binary(image_gray) = convert gray image to binary with GAUSSIAN Threshold\n
    \t\treturn image_object as converted\n
    \t.filter_number(cell) = filter out border table and noise only cell\n
    \t\treturn blank_image_object if null number | image_object_has_only_number\n
    \t.remove_image(image_binary) = remove anything from binary image\n
    \t\treturn blank_image_object\n
    \t.create_contours(image_thresholded) = find all contours in image\n
    \t\treturn list_of_contours_found\n
    \t.adap_threshold(image_gray) = adaptive GAUSSIAN threshold to image\n
    \t\treturn thresholded_image_object\n
    \t.split_cells(image_thresholded) = split image by divide 9x9\n
    \t\treturn np_array 9x9\n
    \t.search_largest_square_contour(list_contours) = find a largest contours in list_contour\n
    \t\treturn contour\n
        """

    def load_image(self):
        return cv2.imread(self.image_path)

    def sudoku_detect(self):
        self.image_gray = cv2.cvtColor(self.load_image(), cv2.COLOR_BGR2GRAY)
        thresh = self.adap_threshold(self.image_gray)
        list_contours = self.create_contours(thresh)
        puzzleCnt = self.search_largest_square_contour(list_contours)

        if puzzleCnt is None:
            raise Exception("Sudoku puzzle not found at " +
                            self.image_path+" path.")
        # result
        self.image_gray = four_point_transform(
            self.image_gray, puzzleCnt.reshape(4, 2))

    def convert_binary(self, image_gray):
        ''' Convert gray image to binary image'''
        image = cv2.fastNlMeansDenoising(image_gray)
        image = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 5)
        return image

    def filter_number(self, cell):
        thresh = cv2.bitwise_not(cell)

        thresh = clear_border(thresh)

        cnts = self.create_contours(thresh)
        mask = np.zeros(thresh.shape, np.uint8)

        # remove all
        if len(cnts) == 0:
            return self.remove_image(thresh)

        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 40:
            return self.remove_image(thresh)

        cv2.drawContours(mask, [c], -1, 255, -1)
        return cv2.bitwise_and(thresh, thresh, mask=mask)

    def remove_image(self, thresh):
        return cv2.bitwise_and(thresh, np.zeros(thresh.shape, np.uint8))

    def create_contours(self, thresh):
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        return imutils.grab_contours(cnts)

    def adap_threshold(self, gray):
        thresh = cv2.GaussianBlur(gray, (7, 7), 3)
        thresh = cv2.adaptiveThreshold(
            thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 5)
        thresh = cv2.bitwise_not(thresh)
        return thresh

    def split_cells(self, image):
        x_size = image.shape[0] // 9
        y_size = image.shape[1] // 9

        array_cell = np.zeros(shape=(9, 9), dtype=object)

        for x in range(0, 9):
            for y in range(0, 9):
                cell = image[x_size *
                             x:x_size * (x+1), y_size * y:y_size * (y+1)]
                cell = self.filter_number(cell)
                array_cell[x][y] = cell
        return array_cell

    def search_largest_square_contour(self, list_contours):
        list_contours = sorted(
            list_contours, key=cv2.contourArea, reverse=True)
        for c in list_contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.06 * peri, True)

            if len(approx) == 4:
                return approx
        return None
