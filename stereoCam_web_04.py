import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify
import threading
import os
import pickle
import open3d as o3d

# Chessboard parameters
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 1.0  # Size of a square in your real world units
# Termination criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Prepare object points
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

app = Flask(__name__)

# Global variables
frame = None
msg = ''
calibrated_left = False
calibrated_right = False
calibration_data = None
point_cloud = None

CALIBRATION_FILE = 'stereo_calibration.pkl'

def save_calibration(data):
    with open(CALIBRATION_FILE, 'wb') as f:
        pickle.dump(data, f)

def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, 'rb') as f:
            return pickle.load(f)
    return None

def calibrate_camera(frame, objpoints, imgpoints):
    global msg
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
   
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners2, ret)
        msg = ''
    else:
        msg = 'ChessNF'
    return frame, ret, objpoints, imgpoints

def stereo_calibrate(objpoints, imgpoints_left, imgpoints_right, img_size):
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        calibration_data['left'][0], calibration_data['left'][1],
        calibration_data['right'][0], calibration_data['right'][1],
        img_size, criteria=criteria, flags=flags)

    return mtx_left, dist_left, mtx_right, dist_right, R, T

def compute_disparity(left_frame, right_frame):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*16,
        blockSize=11,
        P1=8 * 3 * 11**2,
        P2=32 * 3 * 11**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disparity = stereo.compute(left_frame, right_frame).astype(np.float32) / 16.0
    return disparity

def create_point_cloud(disparity, left_frame, Q):
    points = cv2.reprojectImageTo3D(disparity, Q)
    colors = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
    mask = disparity > disparity.min()
    out_points = points[mask]
    out_colors = colors[mask]
    return out_points, out_colors

def capture_frames():
    global frame, msg, calibrated_left, calibrated_right, calibration_data, point_cloud
    cap = cv2.VideoCapture(0)  # Adjust this if necessary for your dual camera setup
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    calibration_data = load_calibration()
    if calibration_data:
        print("Loaded calibration data from file")
        calibrated_left = True
        calibrated_right = True
    else:
        objpoints = []
        imgpoints_left = []
        imgpoints_right = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Split the frame into left and right views
        height, width = frame.shape[:2]
        left_frame = frame[:, :width//2]
        right_frame = frame[:, width//2:]

        if not calibrated_left or not calibrated_right:
            left_frame, ret_left, objpoints, imgpoints_left = calibrate_camera(
                left_frame, objpoints, imgpoints_left)
            right_frame, ret_right, objpoints, imgpoints_right = calibrate_camera(
                right_frame, objpoints, imgpoints_right)

            if len(objpoints) >= 10:  # Calibrate after collecting 10 good frames
                mtx_left, dist_left, mtx_right, dist_right, R, T = stereo_calibrate(
                    objpoints, imgpoints_left, imgpoints_right, left_frame.shape[:2][::-1])
                
                calibration_data = {
                    'left': (mtx_left, dist_left),
                    'right': (mtx_right, dist_right),
                    'R': R,
                    'T': T
                }
                save_calibration(calibration_data)
                print("Stereo calibration completed and saved")
                calibrated_left = True
                calibrated_right = True
        else:
            # Apply undistortion if calibrated
            mtx_left, dist_left = calibration_data['left']
            mtx_right, dist_right = calibration_data['right']
            left_frame = cv2.undistort(left_frame, mtx_left, dist_left)
            right_frame = cv2.undistort(right_frame, mtx_right, dist_right)

            # Compute disparity
            disparity = compute_disparity(left_frame, right_frame)

            # Compute Q matrix for 3D reconstruction
            R = calibration_data['R']
            T = calibration_data['T']
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                mtx_left, dist_left, mtx_right, dist_right,
                left_frame.shape[:2][::-1], R, T)

            # Create point cloud
            points, colors = create_point_cloud(disparity, left_frame, Q)
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            
            # Visualize point cloud (you might want to do this in a separate thread)
            o3d.visualization.draw_geometries([pcd])

        # Combine frames side by side
        frame = np.hstack((left_frame, right_frame))

        if msg != '':
            print(msg)

def gen_frames():
    global frame
    while True:
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stereo Camera Calibration and Point Cloud Generation</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; text-align: center; }
            img { max-width: 100%; height: auto; }
            #status { margin-top: 20px; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Stereo Camera Calibration and Point Cloud Generation</h1>
        <img src="{{ url_for('video_feed') }}" alt="Stereo Camera Feed">
        <div id="status">Status: <span id="calibration-status">Initializing...</span></div>
        <script>
            function updateStatus() {
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('calibration-status').textContent = data.status;
                    });
            }
            setInterval(updateStatus, 1000);
        </script>
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    global calibrated_left, calibrated_right
    if calibrated_left and calibrated_right:
        status = "Calibrated. Generating point cloud."
    else:
        status = "Calibrating cameras. Please show the chessboard pattern."
    return jsonify({"status": status})

if __name__ == '__main__':
    threading.Thread(target=capture_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)