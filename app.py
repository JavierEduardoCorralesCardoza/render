from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Inicia la cámara
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # Lee un fotograma de la cámara
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convierte la imagen de BGR a JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Devuelve el fotograma en formato compatible con el streaming de video
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
