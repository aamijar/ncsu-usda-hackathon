#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import argparse
import time
import sys

def crop_center(img,cropx,cropy):
    y,x,z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx, :]

cam_options = ['rgb', 'left', 'right']

parser = argparse.ArgumentParser()
parser.add_argument("-cam", "--cam_input", help="select camera input source for inference", default='rgb', choices=cam_options)#'models/deeplab_v3_plus_mvn2_decoder_256_openvino_2021.2_6shave.blob'
parser.add_argument("-nn", "--nn_model", help="select camera input source for inference", default='models/4_class_model_mobilenet_v3_large_data4_combined_class_weights_512x512_without_softmax.blob', type=str)
# old model: class_model_mobilenet_v3_small_data3_class_weights_512x512_without_softmax_6shaves.blob

args = parser.parse_args()

cam_source = args.cam_input
nn_path = args.nn_model

nn_shape = 256 #size of square image
if '513' in nn_path:
    nn_shape = 513
if '512' in nn_path:
    nn_shape = 512
def decode_deeplabv3p(output_tensor):
    # ["Soil":BROWN,"Clover:RED","Broadleaf:PURPLE","Grass:ORANGE"]
    class_colors = [[40, 86,166 ], [28, 26,228 ], [184 , 126, 155], [0, 127, 255]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)

    output = output_tensor.reshape(nn_shape,nn_shape)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors

def show_deeplabv3p(output_colors, frame):
    # save output colors.
    return cv2.addWeighted(frame,1, output_colors,0.2,0)


# Start defining a pipeline
pipeline = dai.Pipeline()

if '513' in nn_path:
    nn_shape = 513
    pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)
if '512' in nn_path:
    nn_shape = 512
    pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_3)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(nn_path)

detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

cam=None
# Define a source - color camera
if cam_source == 'rgb':
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(nn_shape,nn_shape)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam.preview.link(detection_nn.input)
elif cam_source == 'left':
    cam = pipeline.createMonoCamera()
    cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
elif cam_source == 'right':
    cam = pipeline.createMonoCamera()
    cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)

if cam_source != 'rgb':
    manip = pipeline.createImageManip()
    manip.setResize(nn_shape,nn_shape)
    manip.setKeepAspectRatio(True)
    manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.out.link(manip.inputImage)
    manip.out.link(detection_nn.input)

#cam.setFps(40)not used in PSA implementation
####################ROI##########################################
stepSize = 0.01
newConfig = False
# Define a source - two mono (grayscale) cameras
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
#spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

xoutDepth = pipeline.createXLinkOut()
#xoutSpatialData = pipeline.createXLinkOut()
#xinSpatialCalcConfig = pipeline.createXLinkIn()

xoutDepth.setStreamName("depth")
#xoutSpatialData.setStreamName("spatialData")
#xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# MonoCamera
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

outputDepth = True
outputRectified = True
lrcheck = True
subpixel = False

# StereoDepth
stereo.setOutputDepth(outputDepth)
stereo.setOutputRectified(outputRectified)
stereo.setConfidenceThreshold(255)
#stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)#Artem
#stereo.setExtendedDisparity(True)


stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
####################Depth Cropping##################
# Crop range
topLeftDepth = dai.Point2f(0, 0) ## X1/720,Y2/1280
bottomRightDepth = dai.Point2f(1, 1) ##X2/720 Y2/1280
#Properties
manip = pipeline.createImageManip()
manip.initialConfig.setCropRect(topLeftDepth.x, topLeftDepth.y, bottomRightDepth.x, bottomRightDepth.y)
manip.setMaxOutputFrameSize(monoRight.getResolutionHeight()*monoRight.getResolutionWidth()*3)
#Linking:
configIn = pipeline.createXLinkIn()
configIn.setStreamName('config')
configIn.out.link(manip.inputConfig)
stereo.depth.link(manip.inputImage)
manip.out.link(xoutDepth.input)

# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("nn_input")
xout_rgb.input.setBlocking(False)

detection_nn.passthrough.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
xout_nn.input.setBlocking(False)

detection_nn.out.link(xout_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)
device.startPipeline()

# Output queues will be used to get the rgb frames and nn data from the outputs defined above
q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False) # Nueral network input
q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False) # Neural network output

depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

start_time = time.time()
counter = 0
fps = 0
layer_info_printed = False

count = 0


while True:
    count +=1
    '''
    inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived

    depthFrame = inDepth.getFrame() # npy array, 720x1280
    # print(depthFrame.shape)
    depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    depthFrameColor = cv2.equalizeHist(depthFrameColor)
    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

    cv2.imshow("depth", depthFrameColor)
    '''

    newConfig = False
    key = cv2.waitKey(1) & 0xFF # Artem
    if key == ord('q'):
        break
    # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
    in_nn_input = q_nn_input.get()
    in_nn = q_nn.get()

    if in_nn_input is not None: ######## Neural Network input
        # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
        shape = (3, in_nn_input.getHeight(), in_nn_input.getWidth())

        frame = in_nn_input.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        # print("RGB input shape:", frame.shape, frame)
    if in_nn is not None: ######Nueral Network Prediciton
        # print("NN received")
        layers = in_nn.getAllLayers()

        if not layer_info_printed:
            for layer_nr, layer in enumerate(layers):
                print(f"Layer {layer_nr}")
                print(f"Name: {layer.name}")
                print(f"Order: {layer.order}")
                print(f"dataType: {layer.dataType}")
                print(f"layer.dimes: {layer.dims}")
                # dims = layer.dims[::-1] # reverse dimensions
                dims = layer.dims
                print(f"dims: {dims}")
            layer_info_printed = True

        # get layer1 data
        layer1 = in_nn.getLayerInt32(layers[0].name)

        # print("layer dims", layer.dims)
        # print("layer", layer)
        lay1 = np.asarray(layer1, dtype=np.int32).reshape(dims)

        output_colors = decode_deeplabv3p(lay1)

        # display seg_mask
        labeled_output = output_colors #New screen for segmentation mask + labels
        cv2.putText(labeled_output, "Soil: BROWN, Clover: RED, Broadleaf: PURPLE , Grass: ORANGE", (2, labeled_output.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.45, (255, 0, 0))
        cv2.imshow("seg_mask",labeled_output)
        cv2.imshow("center crop", crop_center(labeled_output, 200, 200))
        # print("seg mask labeled_output")
        # print(labeled_output.shape)
        if frame is not None:
            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
            # original rgb frame stream display
            cv2.imshow("nn_input", frame)
        # break
    counter+=1
    if (time.time() - start_time) > 1 :
        fps = counter / (time.time() - start_time)

        counter = 0
        start_time = time.time()