#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import time  # for calculateing framerate

BLUE_BGR = [255, 0, 0]
PINK_BGR = [255, 20, 147]
if len(sys.argv) == 2:
    FILE_PATH = sys.argv[1]
elif len(sys.argv) == 1:
    FILE_PATH = "4.png"  # "hand4.png" #customize it!
else:
    print("Too many arguments")
    exit()


def detectHandByColors(image):
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    mask = cv2.inRange(converted, lower, upper)

    kernel11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel11, iterations=2)
    mask = cv2.erode(mask, kernel11, iterations=2)
    mask = cv2.dilate(mask, kernel5, iterations=1)

    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    skin = cv2.bitwise_and(converted, converted, mask=mask)

    image = cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)

    return image


def determineContours(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, image = cv2.threshold(
        image, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    _, contours, hierarchy = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy


def determineBiggestContour(contours):
    maxArea = cv2.contourArea(contours[0])
    biggestContour = contours[0]

    for i in range(1, len(contours)):
        area = cv2.contourArea(contours[i])
        if area > maxArea:
            biggestContour = contours[i]
            maxArea = area

    return biggestContour


def determineConvexityDefects(contour, convexHull):
    defects = cv2.convexityDefects(contour, convexHull)
    defectsList = []

    try:
        for [defect] in defects:
            start, end, farthest, x = defect
            if not(x > 10000):
                continue
            [startPoint], [endPoint], [
                farthestPoint] = contour[start], contour[end], contour[farthest]
            defectsList.append(
                (tuple(startPoint), tuple(endPoint), tuple(farthestPoint)))
    except TypeError as e:
        print("There are no detected defects")

    return defectsList


def calculateCenterMass(contour):
    moments = cv2.moments(contour)
    cx, cy = int(moments['m10'] / moments['m00']
                 ), int(moments['m01'] / moments['m00'])
    return(cx, cy)


def euclideanDistance(x, y):
    x1, x2 = x
    y1, y2 = y
    return np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))


def calculateFramerate(camera):
    fps = camera.get(cv2.CAP_PROP_FPS)
    return fps

def meanOfImages(images):
    images = [np.float32(x) for x in images] # able to hold values above 255
    length = len(images)
    mean = sum(images)/length
    return np.uint8(mean)


def main():

    cap = cv2.VideoCapture(0)
    frameRate = int(calculateFramerate(cap))
    print("Camera's fps: ", frameRate)

    while (cap.isOpened()):

        #    image = cv2.imread(FILE_PATH, cv2.IMREAD_COLOR)

        images = []
        for x in range(frameRate//2):
            _, drawing = cap.read()
            images.append(drawing)
        drawing = meanOfImages(images)
        image = detectHandByColors(drawing)


        contours, hierarchy = determineContours(image)

        biggestContour = determineBiggestContour(contours)
        hull = cv2.convexHull(biggestContour)
        hullNoPoints = cv2.convexHull(biggestContour, returnPoints=False)

        defects = determineConvexityDefects(biggestContour, hullNoPoints)

        fingers = 0
        for defect in defects:
            (a, b, c) = defect
            drawing = cv2.circle(drawing, c, 7, [0, 255, 0], -1)
            fingers += 1

        fingers += 1

        if fingers <= 5:
            print('Number of fingers: ', fingers)
        else:
            print('Number of fingers unknown')

        cv2.drawContours(drawing, [biggestContour], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)

        # po lewej pojawia się obramowanie obszaru wykrytego jako dłoń (czerwony convex hull się jebie)
        # po prawej pojawia się sam obszar wykryty jako dłoń (reszta kolorowana
        # na czarno)
        cv2.imshow("images", np.hstack([drawing, image]))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
