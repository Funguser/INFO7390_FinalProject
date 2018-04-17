# MIT License

# Copyright (c) 2018 ZiyaoQiao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cv2
import os
from matplotlib import pyplot as plt

originPath = '/Users/royn/INFO 7390/INFO-7390-Assignment/HW4/Datas/train'

targetPath = '/Users/royn/INFO 7390/INFO-7390-Assignment/HW4/Datas/contourDetected/'

g = os.walk(originPath)
for path,d,filelist in g:
    for filename in filelist:
        if filename.endswith('jpg'):

            img = cv2.imread(originPath+filename)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
            gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

            # subtract the y-gradient from the x-gradient
            gradient = cv2.subtract(gradX, gradY)
            gradient = cv2.convertScaleAbs(gradient)
            (_, thresh) = cv2.threshold(gradient, 100, 255, cv2.THRESH_BINARY)

            thresh = cv2.dilate(thresh, None, iterations=1)
            thresh = cv2.dilate(thresh, None, iterations=1)
            thresh = cv2.erode(thresh, None, iterations=1)
            thresh = cv2.erode(thresh, None, iterations=1)

            image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                                          cv2.CHAIN_APPROX_SIMPLE)  # use cv2.RETR_TREE to locate and lock the tail
            img = cv2.drawContours(img, contours, -1, (0, 255, 0), 5)

            canny_edges = cv2.Canny(img, 300, 300)
            plt.imshow(canny_edges)

            cv2.imwrite(targetPath + filename, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])