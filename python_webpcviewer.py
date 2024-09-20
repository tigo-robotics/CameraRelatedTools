import cv2
import numpy as np
import open3d as o3d
import os
import traceback
from flask import Flask, render_template_string, jsonify, Response
import json
import math

# Checkerboard dimensions
CHECKERBOARD = (9, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images
objpoints = []
imgpointsL = []
imgpointsR = []

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

app = Flask(__name__)

# Global variables for camera and calibration data
cap = None
calib_data = None

def split_stereo_frame(frame):
    height, width, _ = frame.shape
    half_width = width // 2
    left_img = frame[:, :half_width]
    right_img = frame[:, half_width:]
    return left_img, right_img

def stereo_calibrate_and_save(cap, output_file, num_frames=20):
    collected_frames = 0
    while collected_frames < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        left_img, right_img = split_stereo_frame(frame)
        grayL = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

        if retL and retR:
            objpoints.append(objp)
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            imgpointsL.append(cornersL)
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            imgpointsR.append(cornersR)
            collected_frames += 1
            print(f"Collected frame {collected_frames}/{num_frames}")

        combined_frame = np.hstack((left_img, right_img))
        cv2.imshow("Stereo Calibration", combined_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to stop
            break
    
    cv2.destroyAllWindows()
    
    _, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
    _, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

    retval, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, grayL.shape[::-1], criteria=criteria)

    np.savez(output_file, cameraMatrixL=cameraMatrixL, distCoeffsL=distCoeffsL,
             cameraMatrixR=cameraMatrixR, distCoeffsR=distCoeffsR, R=R, T=T)

    print(f"Stereo calibration data saved to {output_file}")

def load_calibration_data(calibration_file):
    data = np.load(calibration_file)
    return (data['cameraMatrixL'], data['distCoeffsL'],
            data['cameraMatrixR'], data['distCoeffsR'],
            data['R'], data['T'])

def reconstruct_point_cloud(left_img, right_img, calib_data):
    try:
        cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T = calib_data
        
        grayL = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        h, w = grayL.shape[:2]
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, (w, h), R, T)

        stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(grayL, grayR)

        points_3D = cv2.reprojectImageTo3D(disparity, Q)
        
        mask = disparity > disparity.min()
        points = points_3D[mask]
        colors = left_img[mask]

        if len(points) == 0:
            print("No valid points found. Skipping point cloud creation.")
            return None, None

        return points.tolist(), colors.tolist()
    except Exception as e:
        print(f"Error in reconstruct_point_cloud: {str(e)}")
        traceback.print_exc()
        return None, None

def project_3d_to_2d(point, focal_length=500, center_x=320, center_y=240):
    x, y, z = point
    scale = focal_length / (z + focal_length)
    x_2d = x * scale + center_x
    y_2d = y * scale + center_y
    return x_2d, y_2d
def is_valid_point(x, y):
    return not (math.isnan(x) or math.isnan(y))

def save_point_cloud_to_ply(points, colors, file_path='point_cloud.ply'):
    """
    Save a point cloud to a PLY file.

    Args:
        points (list of lists): 3D points.
        colors (list of lists): Corresponding RGB colors.
        file_path (str): Path to the file where the point cloud will be saved.
    """
    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    
    # Set the points and colors
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    # Save to PLY file
    o3d.io.write_point_cloud(file_path, point_cloud)
    print(f"Point cloud saved to {file_path}")


@app.route('/get_point_cloud')
def get_point_cloud():
    global cap, calib_data
    
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Failed to capture frame"})
    
    left_img, right_img = split_stereo_frame(frame)
    points, colors = reconstruct_point_cloud(left_img, right_img, calib_data)
    
    if points is not None and colors is not None:
        # Limit the number of points for better performance
        max_points = 1000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = [points[i] for i in indices]
            colors = [colors[i] for i in indices]
        
        # Project 3D points to 2D
        projected_points = [project_3d_to_2d(p) for p in points]
        
        # Create compact data representation
        compact_data = []
        for (proj_point, color) in zip(projected_points, colors):
            x_2d, y_2d = proj_point
            
            # Filter out invalid points (NaN)
            if is_valid_point(x_2d, y_2d):
                compact_data.append({
                    "x": round(x_2d, 1),
                    "y": round(y_2d, 1),
                    "r": int(color[0]),
                    "g": int(color[1]),
                    "b": int(color[2])
                })
        
        # Save the point cloud as a PLY file from the first frame
        save_point_cloud_to_ply(points, colors, file_path='point_cloud_first_frame.ply')
        
        try:
            json_data = json.dumps({"points": compact_data}, ensure_ascii=False)
            return Response(json_data, mimetype='application/json')
        except Exception as e:
            print(f"Error serializing data: {str(e)}")
            return jsonify({"error": "Failed to serialize point cloud data"})
    else:
        return jsonify({"error": "Failed to create point cloud"})
    

@app.route('/')
def index():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SVG Point Cloud Viewer</title>
        <style>
            body { margin: 0; overflow: hidden; }
            #point-cloud-container { width: 100vw; height: 100vh; }
            #debug-info { position: absolute; top: 10px; left: 10px; color: white; background-color: rgba(0,0,0,0.5); padding: 10px; }
        </style>
    </head>
    <body>
        <svg id="point-cloud-container"></svg>
        <div id="debug-info"></div>
        <script>
            const svg = document.getElementById('point-cloud-container');
            const debugInfo = document.getElementById('debug-info');

            function updatePointCloud() {
                fetch('/get_point_cloud')
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            console.error(data.error);
                            debugInfo.innerText = `Error: ${data.error}`;
                            return;
                        }

                        svg.innerHTML = ''; // Clear previous points
                        data.points.forEach(point => {
                            const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                            circle.setAttribute("cx", point.x);
                            circle.setAttribute("cy", point.y);
                            circle.setAttribute("r", "1");
                            circle.setAttribute("fill", `rgb(${point.r},${point.g},${point.b})`);
                            svg.appendChild(circle);
                        });

                        debugInfo.innerText = `Points: ${data.points.length}`;
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        debugInfo.innerText = `Fetch Error: ${error}`;
                    });

                setTimeout(updatePointCloud, 100); // Update every 1 second
            }

            updatePointCloud();
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

def main():
    global cap, calib_data
    
    try:
        # Initialize the camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Calibrate cameras and save data
        calibration_file = "stereo_calibration_data.npz"
        if not os.path.exists(calibration_file):
            stereo_calibrate_and_save(cap, calibration_file, num_frames=20)
        
        # Load stereo camera calibration data
        calib_data = load_calibration_data(calibration_file)

        # Start the Flask app
        app.run(debug=True)

    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()
    finally:
        if cap is not None:
            cap.release()

if __name__ == "__main__":
    main()