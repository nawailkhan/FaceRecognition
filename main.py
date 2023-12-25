import cv2
import numpy as np
#import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load the pre-trained model
model = MTCNN()

# Define the path to the folder containing known faces
dataset_folder = "C:\\Users\\HP PC\\Desktop\\face detection-PF open ended lab\\dataset"

# Lists to store known faces, names, and encodings
known_faces = []
known_names = []
known_encodings = []

# Iterate over subdirectories (each person's folder)
for person_folder in os.listdir(dataset_folder):
    person_path = os.path.join(dataset_folder, person_folder)

    if os.path.isdir(person_path):
        # Iterate over image files in the person's folder
        for filename in os.listdir(person_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Read the image
                image_path = os.path.join(person_path, filename)
                face = cv2.imread(image_path)

                # Detect the face and get the bounding box
                result = model.detect_faces(face)
                if result:
                    x, y, width, height = result[0]['box']
                    x, y, width, height = x * 2, y * 2, width * 2, height * 2

                    # Crop the face and check if the region is not empty
                    face_crop = face[y:y + height, x:x + width]
                    if not face_crop.size == 0:
                        # Resize the face to 128x128
                        face_resize = cv2.resize(face_crop, (128, 128))

                        # Create the encoding for the face
                        face_encode = np.array(face_resize).flatten()

                        # Append the face, name, and encoding to the lists
                        known_faces.append(face_resize)
                        known_names.append(person_folder)  # Use the folder name as the person's name
                        known_encodings.append(face_encode)

# Access the video camera
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if ret:
        print("Frame captured successfully")

        if ret:
        # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            faces = model.detect_faces(rgb_frame)

            # Draw bounding box
            for face in faces:
                x, y, width, height = face['box']
                cv2.rectangle(frame, (x, y), (x+width, y+height), (0,255,0),4)

        # Detect the faces and get the bounding boxes
        result = model.detect_faces(frame)
        if result:
            print("Face detected")
            for box in result:
                x, y, width, height = box['box']
                x, y, width, height = x * 2, y * 2, width * 2, height * 2

                # Crop the face and check if the region is not empty
                face_crop = frame[y:y + height, x:x + width]
                if not face_crop.size == 0:
                    # Resize the face to 128x128
                    face_resize = cv2.resize(face_crop, (128, 128))

                    # Create the encoding for the face
                    face_encode = np.array(face_resize).flatten()

                    # Find the most similar encoding from the known faces
                    similarities = cosine_similarity(known_encodings, [face_encode])
                    most_similar_index = np.argmax(similarities)

                    # Display the name of the person with the highest similarity
                    cv2.putText(frame, known_names[most_similar_index], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('My Face Recognition Project', frame)

    else:
        print("Error capturing frame")

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
