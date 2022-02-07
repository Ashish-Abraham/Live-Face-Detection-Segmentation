# Live-Face-Detection-Segmentation
Python scripts to detect and segment face using opencv, pixellib. Separate scripts for face detction and instance segmentation using Mask R-CNN.
## Model Used
* https://github.com/matterport/Mask_RCNN/releases/tag/v2.0
* https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
## Working
Detects coordinates of face using built-in cascade classifier in opencv. The image is cropped and passed to state of the art neural networks to mask and segment face and objects.
Takes input in real-time from webcam.
## Use
* Download models from the links above and save to directory Models.
* Download required Python libraries.
* Run the required python script.
