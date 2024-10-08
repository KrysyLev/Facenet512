# face recognition

# IMPORT
import cv2 as cv
import numpy as np
import os
# Import FaceNet and SVM model
from keras_facenet import FaceNet
import pickle
from sklearn.preprocessing import LabelEncoder

# Enable TensorFlow to dynamically allocate GPU memory
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and being used.")
    except RuntimeError as e:
        print(f"Error enabling GPU memory growth: {e}")
else:
    print("No GPU detected, running on CPU.")

# Silence TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# INITIALIZE
facenet = FaceNet()
# faces_embeddings = np.load("face_embeddings_done_4classes.npz")

faces_embeddings = np.load("face_embeddings_done_LFW_30_classes.npz")
# Load encoded labels and SVM model
Y = faces_embeddings["arr_1"]
encoder = LabelEncoder()
encoder.fit(Y)

haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
# predict_model = pickle.load(open("SVM_predict_model.pkl", "rb"))
predict_model = pickle.load(open("SVM_predict_model_lfw.pkl", "rb"))

# Capture video from webcam
cap = cv.VideoCapture(0)

# WHILE LOOP for Face Recognition
while cap.isOpened():
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

    for x, y, w, h in faces:
        img = rgb_img[y : y + h, x : x + w]
        img = cv.resize(img, (160, 160))  # Rescale for FaceNet input
        img = np.expand_dims(img, axis=0)  # Expand dims for FaceNet input

        y_pred = facenet.embeddings(img)  # Get embedding from FaceNet
        face_name = predict_model.predict(y_pred)  # Use SVM model for prediction
        final_name = encoder.inverse_transform(face_name)[0]  # Decode the prediction
        
        # Draw rectangle and name
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv.putText(
            frame,
            str(final_name),
            (x, y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )

    # Show the frame
    cv.imshow("Face Recognition:", frame)

    # Exit on 'q' key
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
