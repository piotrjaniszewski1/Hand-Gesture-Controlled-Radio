#!/usr/bin/env python3

import cv2
import numpy as np
import sys

RED_BGR = [0, 0, 255]
GREEN_BGR = [0, 255, 0]
BLUE_BGR = [255, 0, 0]
PINK_BGR = [255, 20, 147]
MIN_DISTANCE = 10000

CONTOUR_THICKNESS = 2
CIRCLE_RADIUS = 7

THRESHOLD = 70

BRIGHTNESS = 0.7
CONTRAST = 0.5
SATURATION = 0.4
EXPOSURE = 0.3

if len(sys.argv) == 2:
    FILE_PATH = sys.argv[1]
elif len(sys.argv) == 1:
    FILE_PATH = "4.png"  # "hand4.png" #customize it!
else:
    print("Too many arguments")
    exit()


def calculateFramerate(camera):
    fps = camera.get(cv2.CAP_PROP_FPS)
    return fps


def initCapturing(cap):
    cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)
    cap.set(cv2.CAP_PROP_CONTRAST, CONTRAST)
    cap.set(cv2.CAP_PROP_SATURATION, SATURATION)
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

    return cap


def establishBrightness(image, cap):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    values = [x[2] for row in image for x in row]

    cap.set(cv2.CAP_PROP_BRIGHTNESS, np.median(values) / 255)


def removeSuperBrightAreas(image):

    for i in range(4, 40, 4):
        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(image2, 255 - i, 255, cv2.THRESH_BINARY)
        image = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)

    return image

def determineContours(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, image = cv2.threshold(
        image, THRESHOLD, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

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


def determineConvexityDefects(contour, convexHull, min_distance):
    defects = cv2.convexityDefects(contour, convexHull)
    defectsList = []
    try:
        for [defect] in defects:
            start, end, farthest, x = defect
            if not(x > min_distance):
                continue
            [startPoint], [endPoint], [
                farthestPoint] = contour[start], contour[end], contour[farthest]
            defectsList.append(
                (tuple(startPoint), tuple(endPoint), tuple(farthestPoint)))
    except TypeError as e:
        print("There are no detected defects")

    return defectsList


def main():

    cap = cv2.VideoCapture(0)
    cap = initCapturing(cap)
    frameRate = int(calculateFramerate(cap))
    print("Camera's fps: ", frameRate)

    previousDefectCount = 0
    captureCount = 0

    while (cap.isOpened()):

        #image = cv2.imread(FILE_PATH, cv2.IMREAD_COLOR)
        _, drawing = cap.read()
        captureCount += 1

        drawing = removeSuperBrightAreas(drawing)

        if captureCount % (frameRate // 2) == 0:
            captureCount = 0
            establishBrightness(drawing, cap)

        contours, hierarchy = determineContours(drawing)

        biggestContour = determineBiggestContour(contours)
        hull = cv2.convexHull(biggestContour)
        hullNoPoints = cv2.convexHull(biggestContour, returnPoints=False)

        defects = determineConvexityDefects(
            biggestContour, hullNoPoints, MIN_DISTANCE)

        defectCount = 0
        for defect in defects:
            (_, _, c) = defect
            drawing = cv2.circle(drawing, c, CIRCLE_RADIUS, GREEN_BGR, -1)
            defectCount += 1

        defectCount += 1

        if previousDefectCount != defectCount:
            if defectCount <= 6:
                print('The recently detected number of fingers: ', defectCount - 1)
            else:
                print('The number of fingers unknown')

        previousDefectCount = defectCount

        cv2.drawContours(drawing, [biggestContour], 0, tuple(GREEN_BGR), CONTOUR_THICKNESS)
        cv2.drawContours(drawing, [hull], 0, tuple(RED_BGR), CONTOUR_THICKNESS)

        cv2.imshow('output', drawing)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
