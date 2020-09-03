# Live-Face-Mask-Detection
Dectecting whether people are wearing face mask from live feed.

[Click to watch Demo](https://drive.google.com/file/d/1pLER3U_3oyT0GOOtHSVoCzxRYMpR28Tj/view?usp=sharing)

## Contents in this repo:
- main.py (The file to run the live face mask detector from camera feed)
- model.h5 (The CNN model trained to detect face mask)
- face_mask_model_training.ipynb (Notebook file in which the model was trained. Refer it for more details about the dataset and the model used for face mask detection)
- face_classifier FOLDER with file haarcascade_face.xml (pretrained classifier provided by OpenCV for face detection)
- README.md (This README file containing details about the project)

## Dependencies:
- Python2 (Preinstalled in Linux systems)
- Tensorflow version 2.0.0 (GPU: ```sudo pip install tensorflow-gpu==2.0.0``` CPU: ```sudo pip install tensorflow==2.0.0```)
- OpenCV (```sudo pip install opencv-python```)
- Argparse (```sudo pip install argparse```)

## To run the detector:
- Download this repo to your local system and extract the folder.
- Install necessary packages mentioned above
- Open terminal in the extracted repo folder and run ```python2 main.py model.h5``` where model.h5 is the path to the face mask detection model.

