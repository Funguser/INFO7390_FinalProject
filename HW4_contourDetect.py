import cv2
import os
from matplotlib import pyplot as plt

originPath = '/Users/royn/INFO 7390/INFO-7390-Assignment/HW4/Datas/train'

targetPath = '/Users/royn/INFO 7390/INFO-7390-Assignment/HW4/Datas/contourDetected/'

g = os.walk(originPath)
for path,d,filelist in g:
    for filename in filelist:
        if filename.endswith('jpg'):

            img = cv2.imread(filename)

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