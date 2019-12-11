#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import time 

BLUE_BGR = [255, 0, 0]
PINK_BGR = [255, 20, 147]
if len(sys.argv) == 2:
    FILE_PATH = sys.argv[1]
elif len(sys.argv) == 1:
    FILE_PATH = "4.png" 
else:
    print("Too many arguments")
    exit()


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


def detect_hand_by_colors(image):
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


def determine_contours(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, image = cv2.threshold(
        image, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    _, contours, hierarchy = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy


def determine_biggest_contour(contours):
    max_area = cv2.contourArea(contours[0])
    biggest_contour = contours[0]

    for i in range(1, len(contours)):
        area = cv2.contourArea(contours[i])
        if area > max_area:
            biggest_contour = contours[i]
            max_area = area

    return biggest_contour


def determine_convexity_defects(contour, convexHull):
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


def calculate_center_mass(contour):
    moments = cv2.moments(contour)
    cx, cy = int(moments['m10'] / moments['m00']
                 ), int(moments['m01'] / moments['m00'])
    return(cx, cy)


def euclidean_distance(x, y):
    x1, x2 = x
    y1, y2 = y
    return np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))


def calculate_frame_rate(camera):
    fps = camera.get(cv2.CAP_PROP_FPS)
    return fps

def mean_of_images(images):
    images = [np.float32(x) for x in images] # able to hold values above 255
    length = len(images)
    mean = sum(images)/length
    return np.uint8(mean)


def aggregate_images(drawing, cap):
    images = []
    for x in range(frame_rate//2):
        _, drawing = cap.read()
        images.append(drawing)

    drawing = mean_of_images(images)
    return detect_hand_by_colors(drawing)


def determine_defects(contours, drawing):
    biggest_contour = determine_biggest_contour(contours)
    hull = cv2.convexHull(biggest_contour)
    hull_no_points = cv2.convexHull(biggest_contour, returnPoints=False)

    cv2.drawContours(drawing, [biggest_contour], 0, (0, 255, 0), 2)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)

    return biggest_contour, determine_convexity_defects(biggest_contour, hull_no_points)


def count_fingers(defects):
    fingers = 0
    for defect in defects:
        (a, b, c) = defect
        drawing = cv2.circle(drawing, c, 7, [0, 255, 0], -1)
        fingers += 1

    fingers = 1

    if fingers <= 5:
        print('Number of fingers: ', fingers)
    else:
        print('Number of fingers unknown')


def main():
    cap = cv2.VideoCapture(0)
    frame_rate = int(calculate_frame_rate(cap))
    print("Camera's fps: ", frame_rate)

    while (cap.isOpened()):
        _, drawing = cap.read()

        drawing = remove_super_bright_areas(drawing)
        establish_brightness(drawing, cap)
        image = aggregate_images(drawing, cap)
        contours, hierarchy = determine_contours(image)

        defects = determine_defects(contours, drawing)
        count_fingers(defects)

        cv2.imshow("images", np.hstack([drawing, image]))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
