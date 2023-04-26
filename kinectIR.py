import cv2
import mediapipe as mp
from pyk4a import PyK4A, Config, ImageFormat, ColorResolution, FPS, DepthMode
import numpy as np

# Initialize Azure Kinect
k4a = PyK4A(Config(
    color_resolution=ColorResolution.OFF,
    depth_mode=DepthMode.NFOV_2X2BINNED,
    camera_fps=FPS.FPS_30,
    synchronized_images_only=False,
    color_format=ImageFormat.COLOR_BGRA32
))

k4a.open()
k4a.start()

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Main loop
while True:
    capture = k4a.get_capture()
    ir_image = capture.ir
    
 # Normalize IR image using percentile method
    ir_min = np.percentile(ir_image, 1) # Find the closest to darkest pixels
    ir_max = np.percentile(ir_image, 99) # Find the closest to brightest pixels
    ir_image_clipped = np.clip(ir_image, ir_min, ir_max) # Clip the image by remove brigter than ir max and darker than ir min
    ir_image_normalized = cv2.normalize(ir_image_clipped, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # we make the brigthtest 0 and the darkest 255, all the other pixel values are adjusted proportiaonally in between

    # Convert IR image into 3 channels using OpenCV image
    ir_image_3channel = cv2.cvtColor(ir_image_normalized, cv2.COLOR_GRAY2RGB)
    
    # Run MediaPipe Hands on the 8-bit grayscale image
    result = hands.process(ir_image_3channel)
    
    # Draw hand landmarks on the image
    annotated_image = ir_image_3channel.copy()
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display the image
    annotated_image_BGR = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('MediaPipe Hands on IR Image', annotated_image_BGR)
    
    # Exit the loop if needed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

k4a.stop()
k4a.close()
cv2.destroyAllWindows()
hands.close()