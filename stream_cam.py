import cv2
import requests
import time

URL = "http://127.0.0.1:5000"

# Use DirectShow backend on Windows to prevent MSMF sample reader locks
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    # Fallback to default index if DirectShow fails
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam. Ensure no other application (Teams, Zoom, Camera app) is using it.")
    exit(1)

# Set common resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"Streaming live webcam frames to VENTUNO Q via USB ({URL})...")
print("Press 'q' in the preview window to exit.")

frame_id = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame_resized = cv2.resize(frame, (224, 224))
        _, encoded_img = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

        frame_id += 1
        t_start = time.time()
        try:
            resp = requests.post(
                URL,
                data=encoded_img.tobytes(),
                headers={'Content-Type': 'image/jpeg'},
                timeout=3.0
            )
            rtt = (time.time() - t_start) * 1000
            print(f"[Frame {frame_id:04d}] Transmitted -> Status: {resp.status_code} ({rtt:.1f}ms)")
        except requests.exceptions.RequestException as e:
            print(f"[Frame {frame_id:04d}] Transfer failed: {e}")

        cv2.imshow("Webcam Streamer Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.03)

finally:
    cap.release()
    cv2.destroyAllWindows()