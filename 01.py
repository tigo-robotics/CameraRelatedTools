import cv2
import numpy as np


def main():
    # Open the camera
    cap = cv2.VideoCapture(0)


    # Set the width and height
    width, height = 1280, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return


    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        
        # Split the frame into left and right views
        left_view = frame[:, :width//2]
        right_view = frame[:, width//2:]

        
        # Display the original dual view
        cv2.imshow("Dual Camera", frame)

        
        # Press 'ESC' to quit
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

