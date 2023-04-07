import cv2 as cv
import numpy as np
import time


def calc_bounding_rect_v2(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    # Create the landmark_array using a list comprehension
    landmark_array = np.array([
        [min(int(landmark.x * image_width), image_width - 1),
         min(int(landmark.y * image_height), image_height - 1)]
        for landmark in landmarks.landmark
    ], dtype=int)

    # Calculate the bounding rectangle
    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list_v2(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = [
        [min(int(landmark.x * image_width), image_width - 1), 
         min(int(landmark.y * image_height), image_height - 1)]
        for landmark in landmarks.landmark
    ]

    return landmark_point


def pre_process_landmark_v2(landmark_list):
    # Convert to numpy array
    landmark_array = np.array(landmark_list)

    # Convert to relative coordinates
    base_point = landmark_array[0]
    relative_landmark_array = landmark_array - base_point

    # Normalize values
    max_value = np.max(np.abs(relative_landmark_array))
    normalized_landmark_array = relative_landmark_array / max_value

    # Convert to a one-dimensional list
    flattened_landmark_list = normalized_landmark_array.flatten().tolist()

    return flattened_landmark_list


def pre_process_point_history_v2(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    # Convert to numpy array
    point_history_array = np.array(point_history)

    # Convert to relative coordinates
    base_point = point_history_array[0]
    relative_point_history_array = point_history_array - base_point
    relative_point_history_array = relative_point_history_array.astype(np.float64)
    relative_point_history_array[:, 0] /= image_width
    relative_point_history_array[:, 1] /= image_height

    # Convert to a one-dimensional list
    flattened_point_history_list = relative_point_history_array.flatten().tolist()

    return flattened_point_history_list


def fps_actual(frame_count, start_time):
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
        fps_rounded = round(fps, 2)
    return fps_rounded