import cv2
import numpy as np
import os
from datetime import datetime

class StereoVision:
    def __init__(self, left_url, right_url):
        self.left_url = left_url
        self.right_url = right_url
        self.calibration_file = 'stereo_calibration.npz'
        self.calibration_params = None
        
    def connect_cameras(self):
        """Connect to both RTSP streams"""
        self.left_cam = cv2.VideoCapture(self.left_url)
        self.right_cam = cv2.VideoCapture(self.right_url)
        
        if not self.left_cam.isOpened() or not self.right_cam.isOpened():
            raise Exception("Failed to connect to one or both cameras")
            
    def capture_calibration_images(self, num_images=20):
        """Capture images for calibration"""
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        checkerboard = (9, 6)  # adjust based on your checkerboard
        objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
        
        objpoints = []
        imgpoints_left = []
        imgpoints_right = []
        
        print("Capturing calibration images. Press 'c' to capture when checkerboard is visible.")
        count = 0
        
        while count < num_images:
            ret_left, frame_left = self.left_cam.read()
            ret_right, frame_right = self.right_cam.read()
            
            if not ret_left or not ret_right:
                continue
                
            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
            
            cv2.imshow('Left', frame_left)
            cv2.imshow('Right', frame_right)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('c'):
                ret_left, corners_left = cv2.findChessboardCorners(gray_left, checkerboard, None)
                ret_right, corners_right = cv2.findChessboardCorners(gray_right, checkerboard, None)
                
                if ret_left and ret_right:
                    corners_left = cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), criteria)
                    corners_right = cv2.cornerSubPix(gray_right, corners_right, (11,11), (-1,-1), criteria)
                    
                    objpoints.append(objp)
                    imgpoints_left.append(corners_left)
                    imgpoints_right.append(corners_right)
                    
                    count += 1
                    print(f"Captured {count}/{num_images}")
        
        cv2.destroyAllWindows()
        return objpoints, imgpoints_left, imgpoints_right, gray_left.shape[::-1]
        
    def calibrate_cameras(self):
        """Perform stereo camera calibration"""
        if os.path.exists(self.calibration_file):
            print("Loading existing calibration...")
            self.calibration_params = np.load(self.calibration_file)
            return
            
        print("Starting new calibration...")
        objpoints, imgpoints_left, imgpoints_right, img_size = self.capture_calibration_images()
        
        ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            None, None, None, None, img_size,
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            mtx_left, dist_left, mtx_right, dist_right,
            img_size, R, T
        )
        
        self.calibration_params = {
            'mtx_left': mtx_left, 'dist_left': dist_left,
            'mtx_right': mtx_right, 'dist_right': dist_right,
            'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q
        }
        
        np.savez(self.calibration_file, **self.calibration_params)
        print("Calibration completed and saved")
        
    def compute_disparity(self):
        """Compute disparity map from stereo images"""
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16*16,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
        
        while True:
            ret_left, frame_left = self.left_cam.read()
            ret_right, frame_right = self.right_cam.read()
            
            if not ret_left or not ret_right:
                continue
                
            # Convert to grayscale
            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
            
            # Compute disparity
            disparity = stereo.compute(gray_left, gray_right)
            disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            disparity_normalized = np.uint8(disparity_normalized)
            
            # Apply colormap for visualization
            disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
            
            # Show images
            cv2.imshow('Left Camera', frame_left)
            cv2.imshow('Right Camera', frame_right)
            cv2.imshow('Disparity Map', disparity_color)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
        
    def run(self):
        """Main run method"""
        try:
            self.connect_cameras()
            self.calibrate_cameras()
            self.compute_disparity()
        finally:
            self.left_cam.release()
            self.right_cam.release()
            cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    left_url = "rtsp://tester:tester@192.168.1.106:554/stream1"
    right_url = "rtsp://tester:tester@192.168.1.127:554/stream1"
    
    stereo = StereoVision(left_url, right_url)
    stereo.run()