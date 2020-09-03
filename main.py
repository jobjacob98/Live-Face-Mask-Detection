import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2 as cv
import argparse


def main(model_path):
    cam = cv.VideoCapture(0)

    cam.set(3, 1280)
    cam.set(4, 720)

    process_this_frame = True

    # each face image should be resized to 35x35 pixels to fit into the model 
    img_size = 35

    # loading the trained face mask model
    face_mask_model = keras.models.load_model(model_path)

    while True:
        # get each video frame
        ret, frame = cam.read()

        # process only the alternate frames of video to save time
        if process_this_frame:
            # converting frame to gray scale
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)           

            # converting frame to RGB
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Haar Cascade pretrained classifier for face detection
            cascade = cv.CascadeClassifier('face_classifier/haarcascade_face.xml')
            faces_rect = cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5)
            
            # go for face mask detection only if faces are detected in the frame by the Haar Cascade classifier
            if len(faces_rect) != 0:
                faces = []
                for (x, y, w, h) in faces_rect:
                    # crop each face from the whole image
                    face_array = rgb_frame[y:y+h, x:x+w]
                    # resize image to fit the model
                    resized_face_array = cv.resize(face_array, (img_size, img_size))
                    # change the type of array from uint8 to float32 for model compatibility
                    face_array = tf.cast(resized_face_array, tf.float32)
                    
                    faces.append(face_array)

                # convert faces array to numpy array and reshape it to fit the model
                faces = np.array(faces).reshape(-1, img_size, img_size, 3)

                # prediction by the model
                mask_predictions = face_mask_model.predict_classes(faces)

        process_this_frame = not process_this_frame

        if len(faces_rect) != 0:
            for (x, y, w, h), prediction in zip(faces_rect, mask_predictions):
                # class 0 represents With Mask
                if prediction == 0:
                    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)
                    font = cv.FONT_HERSHEY_DUPLEX
                    cv.putText(frame, 'With Mask', (x, (y+h)+22), font, 0.6, (255, 255, 255), 1)
                
                # class 1 represents Without Mask
                elif prediction == 1: 
                    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)
                    font = cv.FONT_HERSHEY_DUPLEX
                    cv.putText(frame, 'Without Mask', (x, (y+h)+22), font, 0.6, (255, 255, 255), 1)

        # displaying the video frames
        cv.imshow('Video', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # passing the path to the trained face mask detection model as a command line argument
    parser = argparse.ArgumentParser()                                  
    parser.add_argument('model_path', type=str, help='path to the trained face mask detection model (required)')

    args = parser.parse_args()

    main(args.model_path)