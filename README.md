# opencv-scratchpad
Random OpenCV scripts

## Camera Server with Google Coral TPU
1. Download and install the Coral TPU installer from Google: https://coral.withgoogle.com/tutorials/accelerator
```
wget http://storage.googleapis.com/cloud-iot-edge-pretrained-models/edgetpu_api.tar.gz
tar xzf edgetpu_api.tar.gz
cd python-tflite-source
bash ./install.sh
```

2. Copy over the python library from the Coral TPU installer
```
cp -a ~/python-tflite-source/edgetpu ~/opencv-scratchpad/
```

3. Download the TensorFlow Lite models and labels from Google
```
mkdir -p models/edge-tpu
curl -O models/edge-tpu/inception_v4_299_quant_edgetpu.tflite https://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/inception_v4_299_quant_edgetpu.tflite
curl -O models/edge-tpu/imagenet_labels.txt http://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/imagenet_labels.txt
curl -O models/edge-tpu/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite http://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite
curl -O models/edge-tpu/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite http://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite
curl -O models/edge-tpu/coco_labels.txt http://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/coco_labels.txt
```

4. Run the camera server
```
workon cv
python ./camera_server.py
```

There are two endpoints:
http://localhost:8080/video_feed - the picamera video image with boxes for detected objects
http://localhost:8080/objects.json - json of the detected objects and the model confidence
