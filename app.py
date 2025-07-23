from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Inisialisasi video capture
camera = cv2.VideoCapture(0) # Gunakan 0 untuk webcam internal

def gen_frames():  
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Ubah ke grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Canny
            canny_edges = cv2.Canny(gray, 100, 200)
            
            # Sobel
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            sobel_x_abs = np.absolute(sobel_x)
            sobel_y_abs = np.absolute(sobel_y)
            sobel_x_8u = np.uint8(sobel_x_abs)
            sobel_y_8u = np.uint8(sobel_y_abs)
            sobel_combined = cv2.bitwise_or(sobel_x_8u, sobel_y_8u)

            # Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_abs = np.absolute(laplacian)
            laplacian_8u = np.uint8(laplacian_abs)

            # Ubah gambar hasil deteksi tepi menjadi 3 channel agar bisa digabung
            canny_colored = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
            sobel_combined_colored = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)
            laplacian_colored = cv2.cvtColor(laplacian_8u, cv2.COLOR_GRAY2BGR)

            # Gabungkan semua gambar menjadi satu
            top_row = np.hstack((frame, canny_colored))
            bottom_row = np.hstack((sobel_combined_colored, laplacian_colored))
            
            # Resize bottom_row agar lebarnya sama dengan top_row
            h, w, _ = top_row.shape
            bottom_row_resized = cv2.resize(bottom_row, (w, int(h * bottom_row.shape[0] / bottom_row.shape[1])))

            # Gabungkan baris atas dan bawah
            combined_frame = np.vstack((top_row, bottom_row_resized))

            ret, buffer = cv2.imencode('.jpg', combined_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index_live.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)