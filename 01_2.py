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

    # Create lists to store the checkerboard images for left and right cameras
    images_left = []
    images_right = []

    # Define the checkerboard size
    checkerboard_size = (9, 6)

    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

    # Termination criteria for cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Split the frame into left and right views
        left_frame = frame[:, :width//2]
        right_frame = frame[:, width//2:]

        # Convert the frames to grayscale
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        # Find the checkerboard corners in both images
        found_left, corners_left = cv2.findChessboardCorners(gray_left, checkerboard_size, None)
        found_right, corners_right = cv2.findChessboardCorners(gray_right, checkerboard_size, None)

        # If found in both images, refine the corners and add to the image lists
        if found_left and found_right:
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            
            images_left.append((gray_left, corners_left))
            images_right.append((gray_right, corners_right))
            
            # Draw the corners on the images for visual feedback
            cv2.drawChessboardCorners(left_frame, checkerboard_size, corners_left, found_left)
            cv2.drawChessboardCorners(right_frame, checkerboard_size, corners_right, found_right)
            
            print(f"Found checkerboard! Total images: {len(images_left)}")

        # Combine left and right frames
        display_frame = np.hstack((left_frame, right_frame))

        # Display the dual view with corner detection
        cv2.imshow("Dual Camera", display_frame)

        # Press 'ESC' to quit, 'c' to calibrate
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break
        elif key == ord('c') and len(images_left) > 10:  # 'c' key and enough images
            break

    cap.release()
    cv2.destroyAllWindows()

    # Check if enough checkerboard images were found
    if len(images_left) > 10 and len(images_right) > 10:
        print("Calibrating cameras...")

        # Prepare object points and image points
        objpoints = [objp for _ in range(len(images_left))]
        imgpoints_left = [corners for _, corners in images_left]
        imgpoints_right = [corners for _, corners in images_right]

        # Calibrate left camera
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            objpoints, imgpoints_left, gray_left.shape[::-1], None, None)

        # Calibrate right camera
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

        # Stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right, dist_right,
            gray_left.shape[::-1], criteria=criteria_stereo, flags=flags)

        print("Calibration results:")
        print("Left camera matrix:", mtx_left)
        print("Left distortion coefficients:", dist_left)
        print("Right camera matrix:", mtx_right)
        print("Right distortion coefficients:", dist_right)
        print("Rotation matrix:", R)
        print("Translation vector:", T)

        # Save calibration results
        np.savez('stereo_calibration.npz', mtx_left=mtx_left, dist_left=dist_left,
                 mtx_right=mtx_right, dist_right=dist_right, R=R, T=T)
        print("Calibration data saved to 'stereo_calibration.npz'")

    else:
        print("Not enough checkerboards found. Cannot calibrate cameras.")

if __name__ == '__main__':
    main()