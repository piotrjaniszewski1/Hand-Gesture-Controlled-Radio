#!/usr/bin/env python3

import cv2
import numpy as np
import sys

BLUE_BGR = [255, 0, 0]
PINK_BGR = [255, 20, 147]
MIN_DISTANCE = 10000

if len(sys.argv) == 2:
    FILE_PATH = sys.argv[1]
elif len(sys.argv) == 1:
    FILE_PATH = "4.png"  # "hand4.png" #customize it!
else:
    print("Too many arguments")
    exit()


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

    cap = cv2.VideoCapture(1)

    while (cap.isOpened()):

        #image = cv2.imread(FILE_PATH, cv2.IMREAD_COLOR)
        _, drawing = cap.read()

        image = drawing

        contours, hierarchy = determineContours(drawing)

        biggestContour = determineBiggestContour(contours)
        hull = cv2.convexHull(biggestContour)
        hullNoPoints = cv2.convexHull(biggestContour, returnPoints=False)

        defects = determineConvexityDefects(
            biggestContour, hullNoPoints, MIN_DISTANCE)

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

        cv2.imshow('output', drawing)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
