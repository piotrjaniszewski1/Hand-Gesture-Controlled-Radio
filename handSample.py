#!/usr/bin/env python3
import cv2
import numpy as np
from skimage.morphology import square

'''
In order to use your camera:
1. Uncomment lines: 48, 50, 53, 71
2. Comment lines: 52, 68
3. Pass a positive value as an argument in line 67
'''
BLUE_BGR = [255, 0, 0]
FILE_PATH = "4.png" # "hand4.png" #customize it!

def customedClosing(image, squareSize, dilationIterations, erosionIterarions):
    image = cv2.dilate(image, square(squareSize), iterations=dilationIterations)
    image = cv2.erode(image, square(squareSize), iterations=erosionIterarions)
    return image

def improveShape(image):
    # checked empirically, undermentioned aproximation correct enough
    image = customedClosing(image, 3, 3, 2)
    image = customedClosing(image, 2, 2, 1)

    return image

def determineContours(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #image = improveShape(image) #sometimes needed

    _, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy

def determineBiggestContour(contours):
    maxArea = 0

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > maxArea:
            biggestContour = contours[i]
            maxArea = area

    return biggestContour

def determineConvexityDefects(contour, convexHull):
    defects = cv2.convexityDefects(contour, convexHull)
    defectsList = []

    for [defect] in defects:
        start, end, farthest, _ = defect
        [startPoint], [endPoint], [farthestPoint] = contour[start], contour[end], contour[farthest]
        defectsList.append((tuple(startPoint), tuple(endPoint), tuple(farthestPoint)))

    return defectsList


def main():

    #cap = cv2.VideoCapture(1)

    #while (cap.isOpened()):

        image = cv2.imread(FILE_PATH, cv2.IMREAD_COLOR)
        #_, image = cap.read()

        contours, hierarchy = determineContours(image)
        drawing = np.zeros(image.shape, np.uint8)

        biggestContour = determineBiggestContour(contours)
        hull = cv2.convexHull(biggestContour)
        hullNoPoints = cv2.convexHull(biggestContour, returnPoints=False)

        defects = determineConvexityDefects(biggestContour, hullNoPoints)

        cv2.drawContours(drawing, [biggestContour], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)

        for (_, _, x) in defects:
            drawing = cv2.circle(drawing, x, 5, BLUE_BGR, -1)

        cv2.imshow('output', drawing)
        cv2.imshow('input', image)

        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            #break

if __name__ == '__main__':
    main()
