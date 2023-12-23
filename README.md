# FaceRecognition
In this code, we will allow the program to detect faces and label them according to the dataset(imagecapture.py) given by the user. We will implement a face recognition system using the MTCNN (Multi-task Cascaded Convolutional Networks) algorithm and cosine similarity. The MTCNN algorithm is a deep learning-based face detection algorithm that can detect faces in images and videos. Cosine similarity is a metric used to measure the similarity between two vectors.

# Table of Contents:
1. Prerequisites
2. Reference
3. Camera Tester(optional)
4. Image Capture
5. Main
6. Conclusion

# Prerequisites:
LIBRARIES:
1. open cv
2. MTCNN
3. numpy
IDE used: VS code

# References:
https://www.educative.io/answers/face-recognition-in-opencv

# Camera Tester(Optional):
Testing camera to ensure the working of webcame in order to proceed with the main program.

1.Importing Libraries: Importing the cv2 module, which is the main module of the OpenCV library.

2. Video Capture Initialization: Initializing the video capture object by calling the VideoCapture() function and passing 0 as the argument. The argument 0 represents the index of the webcam device. If you have multiple webcams connected, you can specify a different index to select a specific webcam.

3. Video Capture Loop: The code then enters a while loop, which continuously captures video frames from the webcam until the loop is terminated.

4. Frame Capture: Inside the loop, the read() function is called on the video capture object to read the next frame from the webcam. The read() function returns two values: ret (a boolean indicating whether the frame was successfully read) and frame (the captured frame).

5. Displaying the Frame: The captured frame is then displayed using the imshow() function, which takes two arguments: the window name (in this case, 'Test Camera') and the frame to be displayed.

6. Terminating the Loop: The code checks for a key press event using the waitKey() function. If the key pressed is 'q' (as specified by ord('q')), the loop is terminated using the break statement.

# Image Capture:
This block of code allows the user to capture images in real-time for the program to use as its dataset. We will use OpenCV to capture images from a webcam and save them to a specified directory. We use 'for loop' to capture 30 images of one person in total and store it in the respective folder. Remember you can increase the dataset for efficient recognition.

1. Importing Libraries: We import the necessary libraries: os for file and directory operations, cv2 for OpenCV functions, time for adding delays, and uuid for generating unique image names.

2. Setting Image directory and number of Images: We set the IMAGES_PATH variable to the directory where we want to save the images. In this example, the images will be saved in the "dataset/nawail" directory. We also specify the number_images variable to determine the number of images we want to capture.

3. Capturing Images: We create a VideoCapture object cap to access the webcam. Inside a loop, we capture images using cap.read() and save them to the specified directory using cv2.imwrite(). Each image is given a unique name using uuid.uuid1(). We display the captured image using cv2.imshow() and add a delay of 0.5 seconds using time.sleep(). The loop continues until the desired number of images is captured or the user presses 'q' to quit. Finally, we release the webcam using cap.release() and close any open windows using cv2.destroyAllWindows().

# Main: 
# KEY CONCEPTS:
1. MTCNN: MTCNN is a deep learning-based face detection algorithm that can detect faces in images and videos. It uses a cascaded architecture of three neural networks to detect faces and facial landmarks.

2. Cosine Similarity: Cosine similarity is a metric used to measure the similarity between two vectors. It calculates the cosine of the angle between the two vectors, which represents their similarity.

# CODE STRUCTURE: 
# 1. Importing the required libraries:

cv2: OpenCV library for image and video processing.
numpy: Library for numerical operations.
mtcnn: Library for implementing the MTCNN algorithm.
sklearn: Library for machine learning algorithms and metrics.
os: Library for interacting with the operating system.

# 2. Loading the pre-trained model:

The MTCNN model is loaded using the MTCNN() function.
Defining the path to the folder containing known faces:

The path to the folder containing the known faces is defined using the dataset_folder variable.

# 3. Initializing lists to store known faces, names, and encodings:

Three lists, known_faces, known_names, and known_encodings, are initialized to store the known faces, corresponding names, and their encodings, respectively.
Iterating over subdirectories (each person's folder):

The code iterates over the subdirectories in the dataset_folder.
For each subdirectory, it iterates over the image files.
If the file ends with ".jpg" or ".png", it reads the image, detects the face using the MTCNN algorithm, crops and resizes the face, and creates an encoding for the face.
The face, name, and encoding are then appended to the respective lists.

# 4. Accessing the video camera and Capturing frames from the video camera:

The code accesses the video camera using the cv2.VideoCapture() function.

The code captures frames from the video camera using the read() function of the video_capture object.
It converts the captured frame to RGB format and detects faces using the MTCNN algorithm.
It draws bounding boxes around the detected faces.

# 5. Finding the most similar encoding from the known faces:

For each detected face, the code creates an encoding for the face.
It calculates the cosine similarity between the encoding of the detected face and the encodings of the known faces.
It finds the index of the most similar encoding using the np.argmax() function.

# 6. Displaying the name of the person with the highest similarity:

The code displays the name of the person with the highest similarity using the cv2.putText() function.

# 7. Displaying the frame:

The code displays the frame with the bounding boxes and the name of the person.
