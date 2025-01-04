import os
import cv2

# Setting up the directory where data will be stored
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):  # Check if the directory exists
    os.makedirs(DATA_DIR)        # If not, create it

# Define the number of classes and dataset size
number_of_classes = 26
dataset_size = 3000

# Initialize video capture from default camera (webcam)
cap = cv2.VideoCapture(0)

for j in range(number_of_classes):

    # Create a directory for each class if it doesn't exist
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))
    print('Collecting data for class {}'.format(j))

    # Collect initial frames until the user is ready (presses 'q')
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam

        # Apply histogram equalization to improve contrast
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for equalization
        equalized_frame = cv2.equalizeHist(gray_frame)
        frame = cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2BGR)  # Convert back to BGR

        # Apply Laplacian transform for edge detection
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)

        # Display text on the frame
        cv2.putText(laplacian, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        
        cv2.imshow('frame', laplacian)  # Display the laplacian frame
        
        if cv2.waitKey(25) == ord('q'):
            break

    # Collect and save the dataset images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()   # Read a frame from the webcam
        # Save the original frame as an image in the respective class directory 
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()