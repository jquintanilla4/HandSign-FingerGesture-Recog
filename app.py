#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from utils import calc_bounding_rect
from utils import calc_landmark_list
from model import KeyPointClassifier
from model import PointHistoryClassifier

keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()
# ADD the keyEvents_action here

# VARIABLE DECLARING
cap_width = 960 # int
cap_height = 540 # int
use_static_image_mode = False
# False is for only working when it recognizes a hand, True is for non-stop looking for a hand in each frame
min_detection_confidence = 0.8 # float
min_tracking_confidence = 0.5 # int
use_brect = True

# MEDIAPIPE MODEL LOAD
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image) # make a copy
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # mediapipe likes
    image.flags.writeable = False  # save memory
    results = model.process(image) # detection
    image.flags.writeable = True  # save memory
    return image, results, debug_image


# READ LABELS
with open('model/keypoint_classifier/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]
with open(
        'model/point_history_classifier/point_history_classifier_label.csv',
        encoding='utf-8-sig') as f:
    point_history_classifier_labels = csv.reader(f)
    point_history_classifier_labels = [
        row[0] for row in point_history_classifier_labels
    ]

# FPS MEASUREMENT ### maybe del
cvFpsCalc = CvFpsCalc(buffer_len=10)

# COORDINATE HISTORY ### maybe del
history_length = 16
point_history = deque(maxlen=history_length)

# FINGER GESTURE HISTORY ### mayb del
finger_gesture_history = deque(maxlen=history_length)

#  Setting the mode to deafult 0 as this is inference mode
mode = 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)  # Usually 0 works for most people, but if it doesn't try 1 or 2 and so on
    args = parser.parse_args()
    return args


# ARGUMENT PASSING
args = get_args()
cap_device = args.device


def select_mode(key, mode): ########################## NOT NECESSARY FOR INFERENCE
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k is for Logging Key Point mode / hand gesture recognition mode
        mode = 1
    if key == 104:  # h is for Logging Point History mode / finger gesture recognition mode
        mode = 2
    return number, mode


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        # you're subtracting the the temp landmark coordinates by the landmarmk points, in this case LANDMARK POINT 0 cordinates
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        # you're subtracting the the temp landmark coordinates by the landmarmk points, in this case LANDMARK POINT 1 cordinates
        # you're subtracting every point in the list(in the hand), hence the list, and changing it to relative coordinates to point 0 and 1

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    # here we're taking values and making them absolute, basically [-1, 2, -3] -> [1, 2, 3], no negative values
    # we then return the biggest item

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    # here n becomes the temp landmark list and gets devided by the max value(biggest item)

    # returns numbers that are between -1 and 1, which is super useful for training.
    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return
    

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


# CAMERA PREP
cap = cv.VideoCapture(cap_device)
cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
# THE LOOP
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        fps = cvFpsCalc.get()

        key = cv.waitKey(10)
        if key == 27: # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture
        ret, frame = cap.read()  # ret rerturns a boolean, so if false it breaks us out
        if not ret:
            break

        image, results, debug_image = mediapipe_detection(frame, hands)
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(debug_image, hand, mp_hands.HAND_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), # joints color
                                        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)) # lines color

        # Getting and collecting the landmarks
        # hands = results.multi_hand_landmarks
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)

                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                # Point gesture (line 3 on the keypoint_classifier_label, base index 2)
                if hand_sign_id == 2:
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                # debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Video feedback window
        cv.imshow('Hand Gesture Recognition', debug_image)

        # if cv.waitKey(10) & 0xFF == ord('q'): # quit the program with the letter q
        #     break

cap.release()
cv.destroyAllWindows()