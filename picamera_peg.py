# import the necessary packages
import sys
import time
from itertools import combinations
from math import ceil

import cv2
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray

from grip import GripPipeline

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = .75
textColor = (255, 255, 255)
textThickness = 2


# Sets up camera for specific use
def camera_setup(camera):
    camera.resolution = (WINDOW_WIDTH, WINDOW_HEIGHT)
    camera.framerate = 32


def validate(c):
    corrected_rectangles = []

    for i in range(len(c)):
        if c[i][2] <= -45:
            corrected_rectangles.append((c[i][1][1], c[i][1][0], c[i][2]))
        else:
            corrected_rectangles.append((c[i][1][0], c[i][1][1], c[i][2]))

    return corrected_rectangles


# Returns from a minAreaRect the width, height and angle
def get_rect_attributes(rectangle):
    return rectangle[0][0], rectangle[0][1], rectangle[2]

def place_text_under(image, sizes, text_list, rect):
    # TODO implement the logic that handles as to not collide with other text, not go out of bounds

    assert len(sizes[1:]) == len(text_list)

    x, y = get_rect_bottom_left_xy(rect)

    for i in range(len(text_list)):
        y += sizes[i + 1][1]

        put_text(image, text_list[i], (x, y))


def set_text_info(image, rect, spacing=2):
    width, height, angle = get_rect_attributes(rect)

    width_str = "Width:{}".format(round(width, 3))
    height_str = "Height:{}".format(round(height, 3))
    angle_str = "Angle:{}".format(round(angle, 3))

    text_list = (width_str, height_str, angle_str)

    sizes = get_stacked_text_size(text_list, spacing)

    place_text_under(image, sizes, text_list, rect)


def get_rect_bottom_left_xy(rect):
    return int(ceil(rect[0][0] - rect[1][0] / 2.0)), int(ceil(rect[0][1] + rect[1][1] / 2.0))


# returns tuple (max_width, total_height)
def get_stacked_text_size(strings, spacing):
    location = [[0, (len(strings) - 1) * spacing]]

    for i in range(len(strings)):
        text = get_text_size(strings[i])

        location[0][0] = max(location[0][0], text[0])
        location[0][1] += text[1] + (spacing if i != len(strings) - 1 else 0)

        location.append(text)

    return location


def add_info_to_contours(bounding_boxes, image):
    for rectangle in bounding_boxes:
        set_text_info(image, rectangle)


def draw_bounding_boxes(contours, image):
    min_area_rectangles = []

    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

        # cv2.circle(image, (int(rect[0][0]) ,int( rect[0][1])), 3, (0, 0, 255))

        min_area_rectangles.append(rect)

    return min_area_rectangles


def get_moments(con):
    moments = []

    for c in con:
        m1 = cv2.moments(c)

        x1 = int(m1["m10"] / m1["m00"])
        y1 = int(m1["m01"] / m1["m00"])

        moments.append((x1, y1))

    return moments


def draw_connection_lines(x_ys, image):
    for pairs in combinations(x_ys, 2):
        cv2.line(image, pairs[0], pairs[1], (0, 255, 0))


def get_area(rectangle):
    return rectangle[1][0] * rectangle[1][1]


def draw_center_weighted_line(x_ys, rectangles, image):
    areas = 0

    center_x = 0

    for i in range(len(rectangles)):
        area = get_area(rectangles[i])

        center_x += x_ys[i][0] * area
        areas += area

    center_x /= (1 if areas == 0 else areas)

    center_x = int(ceil(center_x))

    cv2.line(image, (center_x, 0), (center_x, WINDOW_HEIGHT), (255, 255, 0))
    put_text(image, "CenterX:{}".format(center_x), (3, WINDOW_HEIGHT - 5))

    return center_x


# Returns a tuple (width, height)
def get_text_size(text):
    return cv2.getTextSize(text, fontFace, fontScale, textThickness)[0]


def put_text(image, text, location):
    cv2.putText(image, text, location, fontFace, fontScale, textColor, textThickness)


def draw(c, image):
    cv2.drawContours(image, c, -1, (255, 0, 0), 3)

    bounding_boxes = draw_bounding_boxes(c, image)

    x_ys = get_moments(c)

    draw_connection_lines(x_ys, image)
    center_x = draw_center_weighted_line(x_ys, bounding_boxes, image)

    add_info_to_contours(bounding_boxes, image)

    return center_x


def main(system_arguments):
    # initialize the camera and grab a reference to the raw camera capture
    with PiCamera() as camera:

        camera_setup(camera)
        raw_capture = PiRGBArray(camera, size=(WINDOW_WIDTH, WINDOW_HEIGHT))

        gripped = GripPipeline()

        # allow the camera to warm up
        time.sleep(0.1)

        # capture frames from the camera
        for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            image = frame.array
            gripped.process(image)

            c = gripped.filter_contours_output
            draw(c, image)

            # show the frame
            cv2.imshow("Frame", image)

            # clear the stream in preparation for the next frame
            raw_capture.truncate(0)

            # if the `q` key was pressed, break from the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    # TODO make it so that main uses sys.argv as an argument and the user has to pass in "python filename.py gui" to have the gui while still returning center x
    sys.exit(main(sys.argv))
