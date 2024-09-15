import cv2
import numpy as np

# Chessboard parameters
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 1.0  # Size of a square in your real world units

# Termination criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE
msg = ''
def calibrate_camera(frame, objpoints, imgpoints):
    global msg
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners2, ret)
        msg=''
    else:
        msg='ChessNF'
    return frame, ret, objpoints, imgpoints

def main():
    cap = cv2.VideoCapture(1)  # Adjust this if necessary for your dual camera setup
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    objpoints_left = []
    imgpoints_left = []
    objpoints_right = []
    imgpoints_right = []

    calibrated_left = False
    calibrated_right = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Split the frame into left and right views
        height, width = frame.shape[:2]
        left_frame = frame[:, :width//2]
        right_frame = frame[:, width//2:]

        if not calibrated_left:
            left_frame, ret_left, objpoints_left, imgpoints_left = calibrate_camera(
                left_frame, objpoints_left, imgpoints_left)
            if len(objpoints_left) >= 10:  # Calibrate after collecting 10 good frames
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints_left, imgpoints_left, left_frame.shape[:2][::-1], None, None)
                if ret:
                    calibrated_left = True
                    print("Left camera calibrated")

        if not calibrated_right:
            right_frame, ret_right, objpoints_right, imgpoints_right = calibrate_camera(
                right_frame, objpoints_right, imgpoints_right)
            if len(objpoints_right) >= 10:  # Calibrate after collecting 10 good frames
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints_right, imgpoints_right, right_frame.shape[:2][::-1], None, None)
                if ret:
                    calibrated_right = True
                    print("Right camera calibrated")

        # Combine frames side by side
        combined_frame = np.hstack((left_frame, right_frame))
        cv2.imshow('Dual Camera Calibration', combined_frame)
        print(msg)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break
        elif key == ord('r'):  # Reset calibration
            objpoints_left = []
            imgpoints_left = []
            objpoints_right = []
            imgpoints_right = []
            calibrated_left = False
            calibrated_right = False
            print("Calibration reset")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()