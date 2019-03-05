from flask import Flask, render_template, Response
from object_detect import ObjectDetect

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def captureAndDetect(object_detect):
    while True:
        frame = object_detect.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(captureAndDetect(ObjectDetect()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080', debug=True)
