#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import keyboard
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from utils import calc_bounding_rect
from utils import calc_landmark_list
from utils import pre_process_landmark
from utils import pre_process_point_history
from utils import draw_bounding_rect
from utils import draw_info_text
from utils import draw_point_history
from utils import draw_info
from model import KeyPointClassifier
from model import PointHistoryClassifier


keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()
# ADD the keyEvents_action here

# VARIABLE DECLARING
# mostly opencv related
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

# FPS MEASUREMENT ### DELETE FOR INFERENCE ONLY
cvFpsCalc = CvFpsCalc(buffer_len=10)

# READ LABELS
with open('model/keypoint_classifier/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
with open(
        'model/point_history_classifier/point_history_classifier_label.csv',
        encoding='utf-8-sig') as f:
    point_history_classifier_labels = csv.reader(f)
    point_history_classifier_labels = [row[0] for row in point_history_classifier_labels]

# COORDINATE HISTORY ### DELETE FOR INFERENCE ONLY
history_length = 16
point_history = deque(maxlen=history_length)

# FINGER GESTURE HISTORY ### DELETE FOR INFERENCE ONLY
finger_gesture_history = deque(maxlen=history_length)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)  # Usually 0 works for most people, but if it doesn't try 1 or 2 and so on
    args = parser.parse_args()
    return args


def mediapipe_detection(image, model):
    image = cv.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image) # make a copy
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # mediapipe likes
    image.flags.writeable = False  # save memory
    results = model.process(image) # detection
    image.flags.writeable = True  # save memory
    return image, results, debug_image


mode = 0 #  Setting the mode to deafult 0 as this is inference mode
def select_mode(key, mode): ### DELETE FOR INFERENCE ONLY
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9 ### MAYBE USE A LISTENER FUNCTION
        number = key - 48
    if keyboard.is_pressed('n'):  # n is for inference mode
        mode = 0
    if keyboard.is_pressed('k'):  # k is for Logging Key Point mode / hand gesture recognition mode
        mode = 1
    if keyboard.is_pressed('h'):  # h is for Logging Point History mode / finger gesture recognition mode
        mode = 2
    return number, mode


def logging_csv(number, mode, landmark_list, point_history_list): ### DELETE FOR INFERENCE ONLY
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


# ARGUMENT PASSING
args = get_args()
cap_device = args.device

# CAMERA PREP
cap = cv.VideoCapture(cap_device)
cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
# THE LOOP
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        fps = cvFpsCalc.get()

        key = cv.waitKey(10) ### MODIFY FOR INFERENCE ONLY
        if key == 27: # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture
        ret, frame = cap.read()  # ret rerturns a boolean, so if false it breaks us out
        if not ret:
            break

        # The Hand and Landmark Detection
        image, results, debug_image = mediapipe_detection(frame, hands)
        # Drawing of landmarks
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(debug_image, hand, mp_hands.HAND_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), # joints color
                                        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)) # lines color

        # THE LOOP INSIDE THE LOOP
        # collecting, calculating, classifying, drawing box and text
        if results.multi_hand_landmarks is not None: ## equivalent to sayin if True
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness): # zip lets you iterate over multiple lists
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

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
                most_common_fg_id = Counter(finger_gesture_history).most_common()

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