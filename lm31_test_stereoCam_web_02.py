import cv2
import numpy as np
from flask import Flask, Response, render_template_string
import threading

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

def capture_frames():
    global frame, msg, calibrated_left, calibrated_right
    cap = cv2.VideoCapture(0)  # Adjust this if necessary for your dual camera setup
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    objpoints_left = []
    imgpoints_left = []
    objpoints_right = []
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
        <title>Stereo Camera Calibration</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; text-align: center; }
            img { max-width: 100%; height: auto; }
            #status { margin-top: 20px; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Stereo Camera Calibration</h1>
        <img src="{{ url_for('video_feed') }}" alt="Stereo Camera Feed">
        <div id="status">Calibration Status: <span id="calibration-status">Initializing...</span></div>
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
        status = "Both cameras calibrated"
    elif calibrated_left:
        status = "Left camera calibrated, right camera not calibrated"
    elif calibrated_right:
        status = "Right camera calibrated, left camera not calibrated"
    else:
        status = "Both cameras not calibrated"
    return {"status": status}

if __name__ == '__main__':
    threading.Thread(target=capture_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)