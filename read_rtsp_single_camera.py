import cv2

# RTSP URL (replace with your actual stream link)
rtsp_url = 'rtsp://tigocamera:076519399@192.168.68.61:554/stream1'

# Open video stream
cap = cv2.VideoCapture(rtsp_url)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Display the frame
    cv2.imshow('RTSP Stream', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

#