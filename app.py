#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import numpy as np
import copy
import keyboard
from collections import Counter
from collections import deque
import pydirectinput
import time
import concurrent.futures
# import logging

import cv2 as cv
import mediapipe as mp
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS, ImageFormat

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

# logging.basicConfig(level=logging.DEBUG)

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
# For the TD actions ### TESTING
last_sign = None
last_detection_time = 0 # wait time in seconds
last_detection_off_time = 0 # track the last time the detection was off
last_detection_on_time = 0 # track the last time the detection was on
detection_off_interval = 10 # wait time in seconds
detection_on_interval = 90 # wait time in seconds

# KINECT INSTANCE
# COMMENT OUT THE ABOVE BLOCK WHEN USING A WEBCAM
k4a = PyK4A(Config(color_format=ImageFormat.COLOR_BGRA32,
                   color_resolution=ColorResolution.RES_720P,
                   depth_mode=DepthMode.NFOV_2X2BINNED,
                   camera_fps=FPS.FPS_30))
# we may have add another argument for selecting azure kinect we want to use; device_index=0
# or whichever index is for the camera we need

# MEDIAPIPE MODEL LOAD
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# READ LABELS
with open('model/keypoint_classifier/keypoint_classifier_label.csv',
          encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
with open('model/point_history_classifier/point_history_classifier_label.csv',
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


# Key press function with threading for the TD actions
# using time.sleep() within the loop can block the program from excuting, causing stuttering
# using threading can solve this problem
def press_key(key, press_duration=0.1, release_duration=0.1):
    pydirectinput.keyDown(key)
    time.sleep(press_duration)
    pydirectinput.keyUp(key)
    time.sleep(release_duration)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)


mode = 0 #  Setting the mode to deafult 0 as this is inference mode
def select_mode(mode):
    global k4a
    if keyboard.is_pressed('n'):  # n is for inference mode
        if mode != 0:
            # change the k4a device color resolution to 720p
            k4a._config.color_resolution = ColorResolution.RES_720P
            # change the k4a device depth mode to 2x2 binned
            k4a._config.depth_mode = DepthMode.NFOV_2X2BINNED
            # change the k4a device synchronized images only to true, helps align color and depth images
            k4a._config.synchronized_images_only = True # default, only synced RGB images will be captured, IR images will not
        mode = 0
    if keyboard.is_pressed('k'):  # k is for Logging Key Point mode / hand gesture recognition mode
        mode = 1
    if keyboard.is_pressed('h'):  # h is for Logging Point History mode / finger gesture recognition mode
        mode = 2
    if keyboard.is_pressed('i'):  # i is for infrared inference mode
        if mode != 3:
            # change the k4a device color resolution to off
            k4a._config.color_resolution = ColorResolution.OFF
            # change the k4a device depth mode to nfov unbinned
            k4a._config.depth_mode = DepthMode.NFOV_UNBINNED
            # change the k4a device synchronized images only to true, helps align color and depth images
            k4a._config.synchronized_images_only = False # it ensures that both the RGB and IR images are captured and synced
        mode = 3
    return mode


key_map = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '-': 10, '=': 11,
    'w': 12, 'e': 13, 'r': 14, 't': 15, 'y': 16, 'u': 17, 'o': 18, 'p': 19, '[': 20,
    ']': 21, '\\': 22, 'a': 23, 's': 24, 'd': 25, 'f': 26, 'g': 27, 'j': 28, 'l': 29,
    ';': 30, "'": 31, 'z': 32, 'x': 33, 'c': 34, 'v': 35, 'b': 36, 'm': 37, ',': 38, '.': 39,
    '/': 40, '`': 41
}
excluded_keys = ['q', 'n', 'k', 'h', 'i']
last_key_pressed = ''
last_timestamp = 0
debounce_interval = 0.1 # in seconds
def logging_csv(bingo, mode, landmark_list, point_history_list):
    global last_key_pressed, last_timestamp
    current_timestamp = time.time()
    if current_timestamp - last_timestamp < debounce_interval: # check for debounce interval
        return
    if mode == 0 or mode == 3:
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


# WEB CAMERA PREP, COMMENT OUT WHILST USING AZURE KINECT
# cap = cv.VideoCapture(cap_device)
# cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
# cap.set(cv.CAP_PROP_BUFFERSIZE, buffer_size)

# AZURE KINECT PREP
k4a.open()
k4a.start()

# For FPS counter
frame_count = 0
start_time = time.time()
# THE LOOP
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while True:
    # while cap.isOpened(): # COMMENT OUT WHILST USING AZURE KINECT
        mode = select_mode(mode) # select mode

        # Camera fps
        frame_count += 1
        fps = fps_actual(frame_count, start_time)
        
        # Azure Kinect camera capture (color image)
        if mode == 0:
            capture = k4a.get_capture()
            color_image = capture.color
            # Resize the Azure Kinect color image to the desired size
            color_image = cv.resize(color_image, (cap_width, cap_height))
            # Convert the Azure Kinect image from BGRA to BGR
            color_image = cv.cvtColor(color_image, cv.COLOR_BGRA2BGR)
        # Azure Kinect camera capture (infrared image)
        elif mode == 3:
            capture_i = k4a.get_capture()
            ir_image = capture_i.ir
            # Normalize IR image using percentile method
            ir_min = np.percentile(ir_image, 1) # Find the closest to darkest pixels
            ir_max = np.percentile(ir_image, 99) # Find the closest to brightest pixels
            ir_image_clipped = np.clip(ir_image, ir_min, ir_max) # Clip the image by remove brigter than ir max and darker than ir min
            ir_image_normalized = cv.normalize(ir_image_clipped, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            # we make the brigthtest 0 and the darkest 255, all the other pixel values are adjusted proportiaonally in between

            # Convert IR image into 3 channels using OpenCV image
            ir_image_3channel = cv.cvtColor(ir_image_normalized, cv.COLOR_GRAY2RGB)
        
        # Web camera capture # COMMENT OUT WHILST USING AZURE KINECT
        # ret, frame = cap.read()  # ret rerturns a boolean, so if false it breaks us out
        # if not ret:
        #     break

        # The Hand and Landmark Detection
        # image, results, debug_image = mediapipe_detection(frame, hands) # COMMENT OUT WHILST USING AZURE KINECT
        if mode == 0:
            image, results, debug_image = mediapipe_detection(color_image, hands)
        elif mode == 3:
            image, results, debug_image = mediapipe_detection(ir_image_3channel, hands)

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
                if mode == 0 or mode == 3:
                    current_sign = hand_sign_id
                    # If the current sign is differentt than the last sign, or if it's been 5 seconds since the last detection
                    if current_sign != last_sign or time.time() - last_detection_time >= 5:
                        if hand_sign_id == 0: # base index 0 is the ID for left swipe
                            executor.submit(press_key, 'f')
                            print('left swipe')
                            last_sign = hand_sign_id
                            last_detection_time = time.time()
                        elif hand_sign_id == 1: # base index 1 is the ID for right swipe
                            executor.submit(press_key, 'j')
                            print('right swipe')
                            last_sign = hand_sign_id
                            last_detection_time = time.time()
                        elif hand_sign_id == 3:  # base index 3 is the ID for toggle detection / turn on
                            # if the last time it was turned off is more than 10 seconds or if the last time it was turned on is more than 90 seconds
                            if time.time() - last_detection_off_time >= detection_off_interval or time.time() - last_detection_on_time >= detection_on_interval:
                                executor.submit(press_key, 'g')
                                print('detection ON')
                                last_sign = hand_sign_id
                                last_detection_time = time.time()
                                last_detection_on_time = time.time()
                        elif hand_sign_id == 4: # base index 4 is the ID for blank
                            print('blank')
                            last_sign = hand_sign_id
                            last_detection_time = time.time()
                        elif hand_sign_id == 5: # base index 5 is the ID for the stop / turn off
                            # if the last time it was turned on is more than 90 seconds, or if the last time it was turned off is more than 10 seconds
                            if time.time() - last_detection_on_time >= detection_on_interval or time.time() - last_detection_off_time >= detection_off_interval:
                                executor.submit(press_key, 't')
                                print('detection OFF')
                                last_sign = hand_sign_id
                                last_detection_time = time.time()
                                last_detection_off_time = time.time()


        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, last_key_pressed)

        # Video feedback window
        cv.imshow('Hand Gesture Recognition', debug_image)

        # Exit the loop if needed
        if cv.waitKey(5) & 0xFF == ord('q'): # quit the program with the letter q
            try: # COMMENT OUT THE TRY AND EXCEPT BLOCK WHEN USING A WEBCAM
                k4a.stop()
                k4a.close()
            except Exception as e:
                print('Azure Kinect', e) # handles the error when the Azure Kinect is not connected
            break
            # if stop_and_close_device(): # COMMENT OUT THE TRY AND EXCEPT BLOCK WHEN USING A WEBCAM
            #     break

# Release the webcam capture, close the OpenCV window
# cap.release() # COMMENT OUT WHILST USING AZURE KINECT
cv.destroyAllWindows()