import cv2
import io
import json
import time
import numpy as np

from base_camera import BaseCamera
from picamera.array import PiRGBArray
from picamera import PiCamera
from edgetpu.detection.engine import DetectionEngine
from edgetpu.classification.engine import ClassificationEngine
from PIL import Image, ImageDraw, ImageFont

class Camera(BaseCamera):
    @staticmethod
    def ReadLabelFile(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        ret = {}
        for line in lines:
            pair = line.strip().split(maxsplit=1)
            ret[int(pair[0])] = pair[1].strip()
        return ret

    @staticmethod
    def id_class_name(class_id):
        for key, value in Camera.classNames.items():
            if class_id == key:
                return value

    @staticmethod
    def drawBoxAndLabel(draw, box, label, outline):
        font = ImageFont.truetype("fonts/RobotoMono-Medium.ttf")
        draw.rectangle(box, outline=outline)
        x = box[0]
        y = box[1]
        draw.text((x-1, y-1), label, font=font, fill=outline)
        draw.text((x+1, y-1), label, font=font, fill=outline)
        draw.text((x-1, y+1), label, font=font, fill=outline)
        draw.text((x+1, y+1), label, font=font, fill=outline)
        draw.text((x, y), label, font=font)

    def frames():
        objectEngine = DetectionEngine("models/edge-tpu/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")
        labels = Camera.ReadLabelFile("models/edge-tpu/coco_labels.txt")
        faceEngine = DetectionEngine("models/edge-tpu/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite")
        inceptionObjectEngine = ClassificationEngine("models/edge-tpu/inception_v3_299_quant_edgetpu.tflite")
        inceptionLabels = Camera.ReadLabelFile("models/edge-tpu/imagenet_labels.txt")
        
        width = 1280
        height = 720
        camera = PiCamera()
        camera.resolution = (width, height)
        camera.framerate = 30
        camera.vflip = True
        camera.hflip = True
        camera.meter_mode = "matrix"
        camera.awb_mode = "auto"
        camera.image_denoise = True
        camera.start_preview()
        try:
            while True:
                objectsInFrame = {}
                stream = io.BytesIO()
                camera.capture(stream, format='jpeg')
                stream.seek(0)
                img = Image.open(stream)
                draw = ImageDraw.Draw(img)
                
                for result in inceptionObjectEngine.ClassifyWithImage(img, top_k=10):
                    score = result[1]
                    if score > 0.1:
                        label = inceptionLabels[result[0]]
                        objectsInFrame[label] = str(score)

                detectedObjects = objectEngine.DetectWithImage(img, threshold=0.05, keep_aspect_ratio=True, relative_coord=False, top_k=10)
                for detection in detectedObjects:
                    confidence = detection.score
                    if confidence > .4:
                        label = labels[detection.label_id]
                        objectsInFrame[label] = str(confidence)
                        box = detection.bounding_box.flatten().tolist()
                        Camera.drawBoxAndLabel(draw, box, label, 'red')
                
                detectedFaces = faceEngine.DetectWithImage(img, threshold=0.05, keep_aspect_ratio=True, relative_coord=False, top_k=10)
                for face in detectedFaces:
                    confidence = face.score
                    if confidence > .2:
                        box = face.bounding_box.flatten().tolist()
                        label = "face"
                        objectsInFrame[label] = str(confidence)
                        Camera.drawBoxAndLabel(draw, box, label, 'green')

                objectsInFrame = json.dumps(objectsInFrame)
                output = io.BytesIO()
                img.save(output, format='JPEG')
                yield (objectsInFrame, output.getvalue())
                stream.close()
        finally:
            camera.stop_preview()
