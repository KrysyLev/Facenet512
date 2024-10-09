import cv2
import os

# Load OpenCV's pre-trained Haar Cascade for face detection
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to create a directory for each person
def create_person_folder(person_name):
    path = f"data/dataset/{person_name}"
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Ask the user for a person's name
person_name = input("Enter the person's name: ")
save_path = create_person_folder(person_name)

face_count = 0
max_faces = 5

print(f"Press 'c' to capture face for {person_name}, and 'q' to quit.")

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Draw a rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the video frame with detected faces
    cv2.imshow(f"Face Capture for {person_name}", frame)

    # Wait for user input
    key = cv2.waitKey(1)

    # If 'c' is pressed, capture and save the face
    if key == ord('c'):
        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_image = frame[y:y+h, x:x+w]
            face_filename = f"{save_path}/face_{face_count + 1}.jpg"
            cv2.imwrite(face_filename, face_image)
            print(f"Saved {face_filename}")
            face_count += 1

        if face_count >= max_faces:
            print(f"Captured {max_faces} faces for {person_name}.")
            break

    # Press 'q' to quit
    elif key == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
