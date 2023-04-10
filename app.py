#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import keyboard
from collections import Counter
from collections import deque
import pydirectinput
import time

import cv2 as cv
import mediapipe as mp

from utils import calc_bounding_rect_v2
from utils import calc_landmark_list_v2
from utils import pre_process_landmark_v2
from utils import pre_process_point_history_v2
from utils import fps_actual
from utils import draw_bounding_rect
from utils import draw_info_text
from utils import draw_point_history
from utils import draw_info
from model import KeyPointClassifier
from model import PointHistoryClassifier


keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

# VARIABLE DECLARING
# For OpenCV
cap_device = 1 # int, this can change according to computer and camera
cap_width = 960 # int
cap_height = 540 # int
buffer_size = 1 # int
# For the Bounding Box
use_brect = True
# For the TD actions
last_sign = None
last_detection_time = 0 # wait time in seconds
last_toggle_detection_time = 0 # wait time in seconds
toggle_interval = 30 # wait time in seconds

# MEDIAPIPE MODEL LOAD
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


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

# COORDINATE HISTORY
history_length = 16
point_history = deque(maxlen=history_length)

# FINGER GESTURE HISTORY
finger_gesture_history = deque(maxlen=history_length)


def mediapipe_detection(image, model):
    image = cv.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image) # make a copy
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # mediapipe likes
    image.flags.writeable = False  # save memory
    results = model.process(image) # detection
    image.flags.writeable = True  # save memory
    return image, results, debug_image


mode = 0 #  Setting the mode to deafult 0 as this is inference mode
def select_mode(mode):
    if keyboard.is_pressed('n'):  # n is for inference mode
        mode = 0
    if keyboard.is_pressed('k'):  # k is for Logging Key Point mode / hand gesture recognition mode
        mode = 1
    if keyboard.is_pressed('h'):  # h is for Logging Point History mode / finger gesture recognition mode
        mode = 2
    return mode


key_map = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '-': 10, '=': 11,
    'w': 12, 'e': 13, 'r': 14, 't': 15, 'y': 16, 'u': 17, 'i': 18, 'o': 19, 'p': 20, '[': 21,
    ']': 22, '\\': 23, 'a': 24, 's': 25, 'd': 26, 'f': 27, 'g': 28, 'j': 29, 'l': 30,
    ';': 31, "'": 32, 'z': 33, 'x': 34, 'c': 35, 'v': 36, 'b': 37, 'm': 38, ',': 39, '.': 40,
    '/': 41, '`': 42
}
excluded_keys = ['q', 'n', 'k', 'h']
last_key_pressed = ''
last_timestamp = 0
debounce_interval = 0.1 # in seconds
def logging_csv(bingo, mode, landmark_list, point_history_list):
    global last_key_pressed, last_timestamp
    current_timestamp = time.time()
    if current_timestamp - last_timestamp < debounce_interval: # check for debounce interval
        return
    if mode == 0:
        pass
    if mode == 1 or mode == 2:
        if bingo in key_map and bingo not in excluded_keys:
            key_num = key_map[bingo] # convert the key to a number
            if mode == 1:
                csv_path = 'model/keypoint_classifier/keypoint.csv'
                with open(csv_path, 'a', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([key_num, *landmark_list])
            if mode == 2:
                csv_path = 'model/point_history_classifier/point_history.csv'
                with open(csv_path, 'a', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([key_num, *point_history_list])
            last_key_pressed = bingo
            last_timestamp = current_timestamp # get the current time


# CAMERA PREP
cap = cv.VideoCapture(cap_device)
cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
cap.set(cv.CAP_PROP_BUFFERSIZE, buffer_size)
# For FPS counter
frame_count = 0
start_time = time.time()
# THE LOOP
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        mode = select_mode(mode) # select mode

        # Camera fps
        frame_count += 1
        fps = fps_actual(frame_count, start_time)
        
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
                brect = calc_bounding_rect_v2(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list_v2(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark_v2(landmark_list)
                pre_processed_point_history_list = pre_process_point_history_v2(debug_image, point_history)

                # Write to the dataset file
                keyboard.on_press(lambda event: logging_csv(event.name, mode, pre_processed_landmark_list, pre_processed_point_history_list))

                # HAND SIGN/GESTURE CLASSIFICATION
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                # if hand_sign_id == 2: # base index 2 is the ID for the pointer sign
                if hand_sign_id == 'Not Applicable': # We don't need this at this point in time
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # FINGER GESTURE CLASSIFICATION
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_info_text(debug_image,
                                             brect,
                                             handedness,
                                             keypoint_classifier_labels[hand_sign_id],
                                             point_history_classifier_labels[most_common_fg_id[0][0]])
                
                # Actions part
                # FOR INFERENCE MODE ONLY
                
                if mode == 0:
                    current_sign = hand_sign_id
                    # If the current sign is different than the last sign, or if it's been 5 seconds since the last detection
                    if current_sign != last_sign or time.time() - last_detection_time >= 5:
                        if hand_sign_id == 0: # base index 0 is the ID for left swipe
                            pydirectinput.keyDown('f')
                            time.sleep(0.1)
                            pydirectinput.keyUp('f')
                            time.sleep(0.1)
                            print('left swipe')
                            last_sign = hand_sign_id
                            last_detection_time = time.time()
                        elif hand_sign_id == 1: # base index 1 is the ID for right swipe
                            pydirectinput.keyDown('h')
                            time.sleep(0.1)
                            pydirectinput.keyUp('h')
                            time.sleep(0.1)
                            print('right swipe')
                            last_sign = hand_sign_id
                            last_detection_time = time.time()
                        elif hand_sign_id == 3:
                            if time.time() - last_toggle_detection_time >= toggle_interval: # base index 3 is the ID for toggle detection
                                pydirectinput.keyDown('g')
                                time.sleep(0.1)
                                pydirectinput.keyUp('g')
                                time.sleep(0.1)
                                print('toggle detection')
                                last_sign = hand_sign_id
                                last_detection_time = time.time()
                                last_toggle_detection_time = time.time()
                        elif hand_sign_id == 4: # base index 4 is the ID for blank
                            print('blank')
                            last_sign = hand_sign_id
                            last_detection_time = time.time()
                        elif hand_sign_id == 5: # base index 5 is the ID for the blank_stop
                            print('blank stop')
                            last_sign = hand_sign_id
                            last_detection_time = time.time()


        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, last_key_pressed)

        # Video feedback window
        cv.imshow('Hand Gesture Recognition', debug_image)


        if cv.waitKey(5) & 0xFF == ord('q'): # quit the program with the letter q
            break

cap.release()
cv.destroyAllWindows()