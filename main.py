"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np 
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

from sys import platform

### MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="cam/webcam, path to directory or image or video file.")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default="cam",
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering "
                        "(0.5 by default)"),
    parser.add_argument("-fm", "--frame_metrics", type=bool, default=False,
                        help="Enable print metrics on frames. "
                        "(False by default)"),
    parser.add_argument("-sf", "--show_frame", type=bool, default=False,
                        help="Enable display frames with OpenCV. "
                        "(False by default)")

    return parser


def connect_mqtt():
    ### Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def preprocess(frame, new_shape, n, c):
    	
    new_frame = cv2.resize(frame, (new_shape))
    new_frame = new_frame.transpose((2,0,1))
    new_frame = new_frame.reshape((n, c, new_shape[0], new_shape[1]))
    
    return new_frame

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    ### Initialize any class variables desired ###
    ct = 0
    duration = 0
    frame_ct = 0
    in_frame = False
    predictions = []
    total_ct = 0
    start_frame = 0
    input_formats = [".png",".bmp",".jpg"]
    input_type = None
    image_paths = []
    image_total = 0
    i = 0

    ### OpenCV settings ###
    color = (78, 45, 0) 
    font = cv2.FONT_HERSHEY_COMPLEX 
    font_scale = 0.4 
    org = (10, 200) 
    thickness = 1
    cap = None
    input_stream = None

    ### Initialise the class ###
    infer_network = Network()
    ### Set Probability threshold for detections ###
    prob_threshold = args.prob_threshold

    ### Load the model through `infer_network` ###
    load_ms = infer_network.load_model(args.model, args.device, 1, 1, 2, args.cpu_extension)

    ### Handle the input stream ###
    n, c, h, w = infer_network.get_input_shape()

    try:
        if args.input == 'cam' or args.input == 'webcam' or args.input == '0': ###if single webcam is available
            input_type = 'cam'
            input_stream = 0 ###if RealSense D435 used, set this to 2 for RGB feed
        elif args.input.endswith('.mp4'): ###check for valid video format
            if not(os.path.exists(args.input)):
                assert os.path.isfile(args.input), "Specified input file doesn't exist"
                return 1
            input_type = 'vid'
            input_stream = args.input
        elif filter(args.input.endswith, input_formats) and not os.path.isdir(args.input): ###check for valid image format
            if not(os.path.exists(os.path.abspath(args.input))):
            	assert os.path.isfile(args.input), "Specified input file doesn't exist"
            	return 1
            input_type = 'img'
            input_stream = os.path.abspath(args.input)
        elif os.path.isdir(args.input) and os.path.exists(args.input):
            for f in os.listdir(args.input):
                ext = os.path.splitext(f)[1]
                if ext.lower() not in input_formats:
                     continue
                image_paths.append(os.path.join(os.path.abspath(args.input),f))
            input_type = 'dir'
            image_total = len(image_paths) ###set total of valid images found
            input_stream = image_paths
        else: 
            log.error("Specified input file {} doesn't exist".format(args.input))
    except Exception as e:
        log.error("ERROR: ", e)
    
    if input_type == "dir":
        cap = cv2.VideoCapture(input_stream[i])
        i+=1
    else:
        cap = cv2.VideoCapture(input_stream)
    
    print("To close the application, press 'ESC' with focus on the output window")
    
    ### Loop until stream is over ###
    while cap.isOpened() or (input_type == 'dir' and i < image_total):
        detections = 0 ###reset detections in current frame
        ### Read from the video capture ###
        ret, frame = cap.read()
        if not ret:
        	break
        ### Pre-process the image as needed ###       
        processed_frame = preprocess(frame, (w,h), n, c)
        frame_ct += 1
        ### Start asynchronous inference for specified request ###
        inf_start = time.time()
        infer_network.exec_net(processed_frame)

        ### Wait for the result ###
        if infer_network.wait() == 0:
            inf_end = time.time()
            det_time = inf_end - inf_start

            ### Get the results of the inference request ###
            res = infer_network.get_output()

            ### Extract any desired stats from the results ###
            ### Draw detections
            for obj in res[0][0]:
                class_id, label, conf, xmin, ymin, xmax, ymax = obj
                if conf >= prob_threshold and label == 1: ###check only for person detections (coco label=1)
                    detections = 1
                    width = frame.shape[1]
                    height= frame.shape[0]
                    xmin = int(xmin*width)
                    ymin = int(ymin*height)
                    xmax = int(xmax*width)
                    ymax = int(ymax*height)
                    top_left = (xmin, ymin)
                    bottom_right = (xmax, ymax)
                    color_obj = (0,255,0)
                    frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color_obj, 2)
                    if not in_frame and not ymin < 75: ###no detection in prev. frame
                        start_frame = frame_ct ###start frame count when person first appears
                        ct = 1
                        total_ct += ct

                    in_frame = True

        predictions.append(detections)
             
        if in_frame:

            if detections == 0:
                frames_since_last_ct += 1
                if frames_since_last_ct > 10: 
                    in_frame = False
                    duration = frame_ct - start_frame
                    start_frame = frame_ct
                    ct = 0
                    frames_since_last_ct = 0 ###reset frames since last count
            else: 
                ct = 1
                frames_since_last_ct = 0
    
        if args.frame_metrics:
            frame = cv2.putText(frame, "People in Frame: {}".format(ct), org, font, font_scale, color, thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, "Avg. Duration (in sec): {}".format(duration/10), (org[0],org[1]+15), font, font_scale, color, thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, "Total Counted: {}".format(total_ct), (org[0],org[1]+30), font, font_scale, color, thickness, cv2.LINE_AA)

        ### Calculate and send relevant information on ###
        ### current_count, total_count and duration to the MQTT server ###
        ### Topic "person": keys of "count" and "total" ###
        ### Topic "person/duration": key of "duration" ###
        client.publish("person", json.dumps({"count": detections,"total": total_ct}))
        client.publish("person/duration", json.dumps({"duration": duration/10}))

        ### Send the frame to the FFMPEG server ###
        ### Write an output image if `single_image_mode` ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
    
        if args.show_frame and (input_type == 'img' or input_type == 'dir'):
            frame = cv2.imshow("Detection Results", frame)
            cv2.waitKey(1500) ###delay to show single image
            if input_type == 'dir' and i < image_total:
                cap = cv2.VideoCapture(input_stream[i])
                i+=1
            else:
                cap.release()
                return 0
        elif input_type == 'dir':
            if i < image_total:
                cap = cv2.VideoCapture(input_stream[i])
                i+=1
            else:
                cap.release()
                return 0
        elif args.show_frame:
            frame = cv2.imshow("Detection Results", frame)
        else:
            continue

        key = cv2.waitKey(1)
        if key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    ### Grab command line args ###
    args = build_argparser().parse_args()
    ### Connect to the MQTT server ###
    client = connect_mqtt()
    ### Perform inference on the input stream ###
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()