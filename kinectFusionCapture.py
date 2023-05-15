import cv2 as cv
from pyk4a import PyK4A, Config, ImageFormat, ColorResolution, FPS, DepthMode, transformation
import numpy as np
# import open3d as o3d

cap_device = 1
cap_width = 1280
cap_height = 720
buffer_size = 1

# Initialize Azure Kinect
k4a = PyK4A(Config(
    color_resolution=ColorResolution.RES_720P,
    depth_mode=DepthMode.NFOV_2X2BINNED,
    camera_fps=FPS.FPS_30,
    synchronized_images_only=True,
    color_format=ImageFormat.COLOR_BGRA32))

k4a.open()
k4a.start()

# Get the calibration and create a Transformation object
calibration = k4a.calibration

# Webcam prep
cap = cv.VideoCapture(cap_device)
cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
cap.set(cv.CAP_PROP_BUFFERSIZE, buffer_size)


# Main loop
while True:
# while cap.isOpened():
    capture = k4a.get_capture()
    # ret, frame = cap.read()
    # if not ret:
    #     break

    ir_image = capture.ir
    depth_image = capture.depth
    # color image from webcam
    # color_image = frame

    # Normalize IR image using percentile method
    ir_min = np.percentile(ir_image, 1) # Find the closest to darkest pixels
    ir_max = np.percentile(ir_image, 99) # Find the closest to brightest pixels
    ir_image_clipped = np.clip(ir_image, ir_min, ir_max) # Clip the image by remove brigter than ir max and darker than ir min
    ir_image_normalized = cv.normalize(ir_image_clipped, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    # we make the brigthtest 0 and the darkest 255, all the other pixel values are adjusted proportiaonally in between
    # Convert IR image into 3 channels using OpenCV image
    ir_image_3channel = cv.cvtColor(ir_image_normalized, cv.COLOR_GRAY2BGR)


    # Transform the infrared image to the color camera's coordinate space
    transformed_depth_image = transformation.depth_image_to_color_camera(depth_image, calibration=calibration, thread_safe=True)
    # Normlize the image
    depth_image_normalized = cv.normalize(transformed_depth_image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    # Convert IR image into 3 channels using OpenCV image
    depth_image_3channel = cv.cvtColor(depth_image_normalized, cv.COLOR_GRAY2BGR)

    
    # Display the image
    cv.imshow('IR Feed', ir_image_3channel)
    cv.imshow('Aligned Depth Image', depth_image_3channel)
    # cv.imshow('Color Image', color_image)
    
    # Exit the loop if needed
    if cv.waitKey(5) & 0xFF == ord('q'): # quit the program with the letter q
        try:
            k4a.stop()
            k4a.close()
        except Exception as e:
            print('Azure Kinect', e) # handles the error when the Azure Kinect is not connected
        break

k4a.stop()
k4a.close()
# cap.release()
cv.destroyAllWindows()