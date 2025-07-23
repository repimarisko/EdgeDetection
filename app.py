
from flask import Flask, render_template, Response, request
import cv2
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def gen_frames(method='canny'):  
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed_frame = None

            if method == 'canny':
                processed_frame = cv2.Canny(gray, 100, 200)
            elif method == 'sobel':
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
                sobel_x_abs = np.absolute(sobel_x)
                sobel_y_abs = np.absolute(sobel_y)
                sobel_x_8u = np.uint8(sobel_x_abs)
                sobel_y_8u = np.uint8(sobel_y_abs)
                processed_frame = cv2.bitwise_or(sobel_x_8u, sobel_y_8u)
            elif method == 'laplacian':
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                laplacian_abs = np.absolute(laplacian)
                processed_frame = np.uint8(laplacian_abs)
            else: # Original
                processed_frame = gray

            # Pastikan frame yang diproses adalah 3 channel agar bisa digabung
            if len(processed_frame.shape) == 2:
                processed_frame_colored = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
            else:
                processed_frame_colored = processed_frame

            # Gabungkan gambar asli dan yang sudah diproses
            combined_frame = np.hstack((frame, processed_frame_colored))

            ret, buffer = cv2.imencode('.jpg', combined_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index_interactive.html')

@app.route('/video_feed/<method>')
def video_feed(method):
    return Response(gen_frames(method=method), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
