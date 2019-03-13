#!/usr/bin/env python
from importlib import import_module
import os
from flask import Flask, render_template, Response

from camera_object_detect_edgetpu import Camera

app = Flask(__name__)

camera = Camera()

@app.route('/')
def redirect():
    return redirect("/ai/")

@app.route('/ai/')
def index():
    return render_template('index.html')

def json(camera):
    frame = camera.get_frame()
    yield(frame[0])

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame[1] + b'\r\n')

@app.route('/ai/video_feed')
def video_feed():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ai/objects.json')
def objects_json():
    return Response(json(camera), mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port="8080", threaded=True)
