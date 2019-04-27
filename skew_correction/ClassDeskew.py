""" Deskews file after getting skew angle """
import numpy as np
import matplotlib.pyplot as plt

from skew_correction.ClassSkewDetector import SkewDetector
from skimage import io
from skimage import img_as_ubyte
from skimage.transform import rotate
import cv2


class Deskew:

    def __init__(self, image_file_path=None, add_angle=0, r_angle=0):
        self.skew_correction_output_dir =  'output/skew_corrected'
        self.image_file_path = image_file_path
        self.image_file_name = self.image_file_path.split('/')[-1]
        self.file_prefix = ""
        for file_part in image_file_path.split('.')[:-1]:
            self.file_prefix += file_part
        self.file_extension = "." + image_file_path.split('.')[-1]
        self.r_angle = r_angle
        self.add_angle = add_angle
        self.skew_obj = SkewDetector(self.image_file_path)

    def deskew(self):
        img = io.imread(self.image_file_path)
        res = self.skew_obj.determine_skew()
        angle = res['Estimated Angle']

        if angle >= 0 and angle <= 90:
            rot_angle = angle - 90 + self.r_angle
        if angle >= -45 and angle < 0:
            rot_angle = angle - 90 + self.r_angle
        if angle >= -90 and angle < -45:
            rot_angle = 90 + angle + self.r_angle

        if abs(rot_angle) > 0.5:
            if rot_angle < 0:
                self.add_angle = self.add_angle * (-1)
            print("The skewed angle is big than 0.5 degree, we need to tune")
            rotated = rotate(img, rot_angle + self.add_angle, resize=True, cval=1, mode='constant')
            io.imsave(self.skew_correction_output_dir + "/" + self.image_file_name, rotated)
            return True
        else:
            print("The skewed angle is less than 0.5 degree, we DO NOT need to tune")
            return False
