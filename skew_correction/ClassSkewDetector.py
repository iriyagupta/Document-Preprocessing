""" Calculates skew angle """
import os
import imghdr
import optparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.transform import hough_line, hough_line_peaks


class SkewDetector:

    piby4 = np.pi / 4

    def __init__(
        self,
        image_file_path,
        batch_path=None,
        output_file=None,
        sigma=3.0,
        display_output=None,
        num_peaks=20,
        plot_hough=None
    ):

        self.sigma = sigma
        self.image_file_path = image_file_path
        self.batch_path = batch_path
        self.output_file = output_file
        self.display_output = display_output
        self.num_peaks = num_peaks
        self.plot_hough = plot_hough


    def get_max_freq_elem(self, arr):

        max_arr = []
        freqs = {}
        for i in arr:
            if i in freqs:
                freqs[i] += 1
            else:
                freqs[i] = 1

        sorted_keys = sorted(freqs, key=freqs.get, reverse=True)
        max_freq = freqs[sorted_keys[0]]

        for k in sorted_keys:
            if freqs[k] == max_freq:
                max_arr.append(k)

        return max_arr


    def compare_sum(self, value):
        if value >= 44 and value <= 46:
            return True
        else:
            return False


    def calculate_deviation(self, angle):

        angle_in_degrees = np.abs(angle)
        deviation = np.abs(SkewDetector.piby4 - angle_in_degrees)
        return deviation


    def check_path(self, path):

        if os.path.isabs(path):
            full_path = path
        else:
            full_path = os.getcwd() + os.sep + str(path)
        return full_path

    def determine_skew(self):
        img = io.imread(self.image_file_path, as_grey=True)
        edges = canny(img, sigma=self.sigma)
        h, a, d = hough_line(edges)
        _, ap, _ = hough_line_peaks(h, a, d, num_peaks=self.num_peaks)

        if len(ap) == 0:
            return {"Message": "Bad Quality"}

        absolute_deviations = [self.calculate_deviation(k) for k in ap]
        average_deviation = np.mean(np.rad2deg(absolute_deviations))
        ap_deg = [np.rad2deg(x) for x in ap]

        bin_0_45 = []
        bin_45_90 = []
        bin_0_45n = []
        bin_45_90n = []

        for ang in ap_deg:

            deviation_sum = int(90 - ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_45_90.append(ang)
                continue

            deviation_sum = int(ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_0_45.append(ang)
                continue

            deviation_sum = int(-ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_0_45n.append(ang)
                continue

            deviation_sum = int(90 + ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_45_90n.append(ang)

        angles = [bin_0_45, bin_45_90, bin_0_45n, bin_45_90n]
        lmax = 0

        for j in range(len(angles)):
            l = len(angles[j])
            if l > lmax:
                lmax = l
                maxi = j

        if lmax:
            ans_arr = self.get_max_freq_elem(angles[maxi])
            ans_res = np.mean(ans_arr)

        else:
            ans_arr = self.get_max_freq_elem(ap_deg)
            ans_res = np.mean(ans_arr)

        data = {
                "Average Deviation from pi/4": average_deviation,
                "Estimated Angle": ans_res,
                "Angle bins": angles
            }

        return data

