import cv2
import mediapipe as mp
from pyk4a import PyK4A, Config, ImageFormat, ColorResolution, FPS, DepthMode, transformation, Calibration
import numpy as np

# Initialize Azure Kinect
k4a = PyK4A(Config(
    color_resolution=ColorResolution.RES_720P,
    depth_mode=DepthMode.NFOV_2X2BINNED,
    camera_fps=FPS.FPS_30,
    synchronized_images_only=True,
    color_format=ImageFormat.COLOR_BGRA32
))

k4a.open()
k4a.start()

# # Get the calibration and create a Transformation object
calibration = k4a.calibration
# cali = Calibration.transformation_handle(calibration)
# transform = transformation(calibration)

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Main loop
while True:
    capture = k4a.get_capture()
    depth_image = capture.depth
    color_image = capture.color

    # Transform the infrared image to the color camera's coordinate space
    transformed_depth_image = transformation.depth_image_to_color_camera(depth_image, calibration=calibration, thread_safe=True)

    # Normlize the image
    depth_image_normalized = cv2.normalize(transformed_depth_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Convert IR image into 3 channels using OpenCV image
    depth_image_3channel = cv2.cvtColor(depth_image_normalized, cv2.COLOR_GRAY2RGB)
    
    # Run MediaPipe Hands on the 8-bit grayscale image
    result = hands.process(depth_image_3channel)
    
    # Draw hand landmarks on the image
    annotated_image = depth_image_3channel.copy()
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display the image
    annotated_image_BGR = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    depth_image_3channel_BGR = cv2.cvtColor(depth_image_3channel, cv2.COLOR_RGB2BGR)
    cv2.imshow('Annotated Hands', annotated_image_BGR)
    cv2.imshow('Aligned Depth Image', depth_image_3channel_BGR)
    cv2.imshow('Color Image', color_image)
    
    # Exit the loop if needed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

k4a.stop()
k4a.close()
cv2.destroyAllWindows()
hands.close()