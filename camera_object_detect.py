import cv2
import time
from base_camera import BaseCamera
from picamera.array import PiRGBArray
from picamera import PiCamera


class Camera(BaseCamera):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Pretrained classes in the model
    classNames = {0: 'background',
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
        7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
        13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
        18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
        24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
        32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
        37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
        41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
        46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
        51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
        56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
        61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
        67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
        75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
        80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
        86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
 

    @staticmethod
    def id_class_name(class_id):
        for key, value in Camera.classNames.items():
            if class_id == key:
                return value

    def frames():
        faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
        model = cv2.dnn.readNetFromTensorflow('models/frozen_inference_graph.pb', 'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
        camera = PiCamera()
        camera.resolution = (1280, 720)
        camera.framerate = 2
        camera.vflip = True
        camera.hflip = True
        camera.meter_mode = "matrix"
        camera.awb_mode = "auto"
        #camera.exposure_mode = "night"
        camera.image_denoise = True
        rawCapture = PiRGBArray(camera, size=(1280, 720))
        time.sleep(0.1)
        #cap = cv2.VideoCapture(0)
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        #cap.set(cv2.CAP_PROP_FPS, 1)

        #while True:
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            image = frame.array
            #image = cv2.cvtColor(frame.array, cv2.COLOR_BGR2GRAY)
            #image_gray = image
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #ret, image = cap.read()
            #image = cv2.imread("image.jpeg")
            image_height, image_width, _ = image.shape
            
            faces = faceCascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

            model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
            output = model.forward()

            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            for detection in output[0, 0, :, :]:
                confidence = detection[2]
                if confidence > .4:
                    class_id = detection[1]
                    class_name=Camera.id_class_name(class_id)
                    print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
                    box_x = detection[3] * image_width
                    box_y = detection[4] * image_height
                    box_width = detection[5] * image_width
                    box_height = detection[6] * image_height
                    cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
                #cv2.putText(image, class_name, (int(box_x), int(box_y+.05*image_height)), cv2.FONT_HERSHEY_SIMPLEX, (.005*image_width),(0, 0, 255))
                    cv2.putText(image, class_name + " " + str(confidence), (int(box_x), int(box_y+.05*image_height)), Camera.font, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
            rawCapture.truncate(0)
            yield cv2.imencode('.jpg', image)[1].tobytes()
