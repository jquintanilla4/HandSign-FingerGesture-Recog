# Hand Sign and Finger Gesture Recognition
**+Windows key events**

<br> This is a fork (2023) of an English translated [fork](https://github.com/kinivi/hand-gesture-recognition-mediapipe) by Nikita Kiselov of the [original repo](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe) by Kazuhito Takahashi.
<br>

## Changes from the original fork
* Simplified the code
* Refactored the code for easier development (well for me at least)
* Annotated the code further
* Added the ability to record more hand signs
* Added the ability to use an Azure Kinect
* Added an infrared mode
* Modified the keypoint_classification_EN.ipynb with more English
* Modified the keypoint_classification_EN.ipynb to add more layers in the model training
* ... and other small additions here and there

## Quick Note
The classification doesn't work well on IR images. I'm working on a new model to fix that issue, so it works in low light and dark conditions.

## Requirements (tested in 2023)
* Tensorflow 2.9.0
* CUDA Toolkit 11.0
* cuDNN 8.1.1
* Python 3.10.6 (tested, but might work on other version)

### app.py
This script is for inference and data collection.

### keypoint_classification.ipynb
This is a model training script for hand sign recognition.

### point_history_classification.ipynb
This is a model training script for finger gesture recognition.


#### Model training
Open "[keypoint_classification.ipynb](keypoint_classification.ipynb)" in Jupyter Notebook and execute from top to bottom.<br>
To change the number of training data classes, change the value of "NUM_CLASSES = 3" <br>and modify the label of "model/keypoint_classifier/keypoint_classifier_label.csv" as appropriate.<br><br>

Open "[point_history_classification.ipynb](point_history_classification.ipynb)" in Jupyter Notebook and execute from top to bottom.<br>
To change the number of training data classes, change the value of "NUM_CLASSES = 4" and <br>modify the label of "model/point_history_classifier/point_history_classifier_label.csv" as appropriate. <br><br>


## Reference
* [MediaPipe](https://mediapipe.dev/)

## Author
Kazuhito Takahashi(https://twitter.com/KzhtTkhs)

## Translation and other improvements
Nikita Kiselov(https://github.com/kinivi)

## Further changes
J. Quintanilla(https://github.com/jquintanilla4)
 
## License 
hand-gesture-recognition-using-mediapipe is under [Apache v2 license](LICENSE).