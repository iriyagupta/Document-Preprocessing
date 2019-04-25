import sys, io
import os
import cv2
import numpy as np
from skew_correction.ClassDeskew import Deskew

class DocumentPreprocessor:
    def __init__(self, image_file_path):
        self.image_file_path = image_file_path
        self.image_file_name = self.image_file_path.split('/')[-1]
        self.base_output_dir = 'output'
        self.skew_correction_output_dir = self.base_output_dir + '/skew_corrected'
        self.black_filtered_dir = self.base_output_dir + '/black_filtered'
        self.greyscaled_dir = self.base_output_dir + '/greyscaled'
        self.gaussian_thresholded_dir = self.base_output_dir + '/gaussian_thresholded'
        self.dialated_dir = self.base_output_dir + '/dialated'
        try:
            os.mkdir(self.base_output_dir)
        except Exception as e:
            print(e)
        try:
            os.mkdir(self.skew_correction_output_dir)
        except Exception as e:
            print(e)
        try:
            os.mkdir(self.black_filtered_dir)
        except Exception as e:
            print(e)
        try:
            os.mkdir(self.greyscaled_dir)
        except Exception as e:
            print(e)
        try:
            os.mkdir(self.gaussian_thresholded_dir)
        except Exception as e:
            print(e)
        try:
            os.mkdir(self.dialated_dir)
        except Exception as e:
            print(e)

        self.skew_corrected_file_path = self.image_file_path
        self.file_prefix = ""
        for file_part in image_file_path.split('.')[:-1]:
            self.file_prefix += file_part
        self.file_extension = "." + image_file_path.split('.')[-1]
        self.cv2_image = cv2.imread(self.image_file_path)

    def perform_skew_correction(self):
        deskew_obj = Deskew(self.image_file_path)
        corrected_flag = deskew_obj.deskew()
        if corrected_flag:
            self.skew_corrected_file_path = self.skew_correction_output_dir + "/" + self.image_file_name
            self.cv2_image = cv2.imread(self.skew_corrected_file_path)

    def filter_black_color_from_image(self, in_place=True, lower_hue=np.array([0,0,0]), upper_hue=np.array([120,120,120])):
        #lower_hue = np.array([0,0,0])
        #upper_hue = np.array([255,255,20])
        hsv = cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hue, upper_hue)
        out_image = 255 - mask
        if in_place:
            self.cv2_image = out_image
        cv2.imwrite(self.black_filtered_dir + "/" + self.image_file_name, out_image)
        return out_image

        

    def perform_image_dialation(self, in_place=True):
        dilate_kernel = 3
        dialate_iter = 1
        #gray_image = cv2.cvtColor(self.cv2_image, cv2.COLOR_RGB2GRAY)
        reversed_gray = cv2.bitwise_not(self.cv2_image)
        dialated_image = cv2.dilate(reversed_gray, np.ones((dilate_kernel,dilate_kernel),np.uint8), iterations=dialate_iter)
        dialated_image = cv2.bitwise_not(dialated_image)
        cv2.imwrite(self.dialated_dir + "/" + self.image_file_name, dialated_image)
        if in_place:
            self.cv2_image = dialated_image
        return dialated_image

    def convert_image_to_grayscale(self, in_place=True):
        in_img = cv2.medianBlur(self.cv2_image, 3)
        gray_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(self.greyscaled_dir + "/" + self.image_file_name, gray_img)
        if in_place:
            self.cv2_image = gray_img
        return gray_img

    def gaussian_thresholding_of_image(self, in_place=True):
        thresholded = cv2.adaptiveThreshold(self.cv2_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        cv2.imwrite(self.gaussian_thresholded_dir + "/" + self.image_file_name, thresholded)
        if in_place:
            self.cv2_image = thresholded
        return thresholded

def main():
    folder_name = 'data'
    for file_name in os.listdir(folder_name):
        print(file_name)
        if 'DS_Store' in file_name:
            continue
        img_file = os.path.join(folder_name, file_name)
        doc_preprocessor = DocumentPreprocessor(img_file)
        doc_preprocessor.perform_skew_correction()
        doc_preprocessor.filter_black_color_from_image()
        #doc_preprocessor.gaussian_thresholding_of_image()
        doc_preprocessor.perform_image_dialation()
    #doc_preprocessor.convert_image_to_grayscale()
    #doc_preprocessor.remove_background_noise()


if __name__ == "__main__":
    main()