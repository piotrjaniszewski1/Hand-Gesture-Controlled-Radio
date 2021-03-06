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


def calculate_frame_rate(camera):
    fps = camera.get(cv2.CAP_PROP_FPS)
    return fps


def init_capturing(cap):
    cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)
    cap.set(cv2.CAP_PROP_CONTRAST, CONTRAST)
    cap.set(cv2.CAP_PROP_SATURATION, SATURATION)
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

    return cap


def establish_brightness(image, cap):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    values = [x[2] for row in image for x in row]

    cap.set(cv2.CAP_PROP_BRIGHTNESS, np.median(values) / 255)


def remove_super_bright_areas(image):

    for i in range(4, 40, 4):
        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(image2, 255 - i, 255, cv2.THRESH_BINARY)
        image = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)

    return image

def determine_contours(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, image = cv2.threshold(
        image, THRESHOLD, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    _, contours, hierarchy = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy


def determine_biggest_contour(contours):
    maxArea = cv2.contourArea(contours[0])
    biggest_contour = contours[0]

    for i in range(1, len(contours)):
        area = cv2.contourArea(contours[i])
        if area > maxArea:
            biggest_contour = contours[i]
            maxArea = area

    return biggest_contour


def determine_convexity_defects(contour, convexHull, min_distance):
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


def determine_defects(contours, previous_defect_count, drawing):
    biggest_contour = determine_biggest_contour(contours)
    hull = cv2.convexHull(biggest_contour)
    hull_no_points = cv2.convexHull(biggest_contour, returnPoints=False)

    defects = determine_convexity_defects(
    biggest_contour, hull_no_points, MIN_DISTANCE)

    defect_count = 0
    for defect in defects:
        (_, _, c) = defect
        drawing = cv2.circle(drawing, c, CIRCLE_RADIUS, GREEN_BGR, -1)
        defect_count += 1

    defect_count += 1

    if previous_defect_count != defect_count:
        if defect_count <= 6:
            print('The recently detected number of fingers: ', defect_count - 1)
        else:
            print('The number of fingers unknown')

    cv2.drawContours(drawing, [biggest_contour], 0, tuple(GREEN_BGR), CONTOUR_THICKNESS)
    cv2.drawContours(drawing, [hull], 0, tuple(RED_BGR), CONTOUR_THICKNESS)

    cv2.imshow('output', drawing)

    return defect_count


def adjust_brightness(drawing, frame_rate, cap):
    drawing = remove_super_bright_areas(drawing)

    if capture_count % (frame_rate // 2) == 0:
        capture_count = 0
        establish_brightness(drawing, cap)

    return drawing

def main():
    cap = cv2.VideoCapture(0)
    cap = init_capturing(cap)
    frame_rate = int(calculate_frame_rate(cap))

    previous_defect_count = 0
    capture_count = 0

    while (cap.isOpened()):
        _, drawing = cap.read()
        capture_count += 1

        drawing = adjust_brightness(drawing, frame_rate, cap)
        contours, hierarchy = determine_contours(drawing)
        previous_defect_count = determine_defects(contours, previous_defect_count, drawing)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
