# Project Write-Up

For this project I decided to use ssdlite_mobilenet_v2_coco_2018_05_09 model as the model footprint was small and for this specific use case it met the project needs. To run the application follow these steps from main directory:

  1. Setup environment variables: $ source /opt/intel/openvino/bin/setupvars.sh
  2. Start the Mosca server, if successful you will see 'Mosca server started': 
      $ cd webservice/server/node-server
      $ node ./server.js
  3. Open new terminal and start the GUI, if successful you will see webpack "Compiled successfully":
      $ cd webservice/ui
      $ npm run dev
  4. Open new terminal and start FFmpeg Server:
      $ sudo ffserver -f ./ffmpeg/server.conf
  5. Open new terminal and run the code:
      $ source /opt/intel/openvino/bin/setupvars.sh
      $ python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m model/ssdlite_mobilenet_v2_coco_2018_05_09/ssdlite_mobilenet_v2_coco.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

## Things to Note:
For an unknown reason the GUI wasn't displaying the data sent to the Mosca server, although data was being sent to it by the client and published. This seemed to work fine and display on my workspace's GUI, either way a Boolean flag was implemented on app to display metrics in frame (-fm, --frame_metrics). I later on realized issue was because by default the webservice is setup to utilize the classroom workspace, so this has been changed. 

Another small bug was the display of the video looked wrong, probably because of a format issue (purple-tint look on video) but OpenCV does display the video correctly in local machine. A feature to display the frame with OpenCV was added (-sf, --show_frame). This issue also applies to webcam feed and single/multiple images as input stream for the application, so the feature to display the frame with OpenCV was helpful in case this issue appears.


## Explaining Custom Layers

The process behind converting custom layers involves implementing functions to extend the Model Optimizer as well as the Inference Engine.

Some of the potential reasons for handling custom layers are to add support for layers not supported by the toolkit for a framework (i.e. TF, Caffe, ONNX, etc.) or target device (i.e. CPU, GPU, VPU, etc.) of choice.
It could be that your custom trained model may have layers that are all supported by the CPU, but some of these layers may not work on GPU/VPU. In this case if you want your model to run on GPU/VPU, as a user you must implement these custom layers to extend this functionality to the Model Optimizer and Inference Engine. 


## Comparing Model Performance

My method to compare models before and after conversion to Intermediate Representations was:
  
  1. Setup environment locally. Hardware specifications and configuration:
      - Intel® NUC Kit NUC8i7HVK with Intel(R) Core(TM) i7-8809 CPU
      - 16 GB of RAM
      - Ubuntu 18.04 LTS
      - TensorFlow 1.15.4
      - Python 3.6.9
      - Intel® Distribution of OpenVINO™ toolkit R3.1 (2019.3.376)

  2. Establish ground truth for object detection
      - Captured each individual frame from the video input and classified them in separate folders according to the object of interest, in our case a person. This was established as my ground truth as these frames could be easily classified by the human eye on whether a person was present in a frame or not. My data ended with a set of two folders; a folder of frames with people (1050 items) and another folder of frames without people (344 items). For a total of 1394 frames in original video. Frames can be found under their respective folder in resources/data. The script is export_frames.py under resources/ and it takes video file as argument.
      - Using Jupyter Notebook I reused a public tutorial available in TensorFlow git repository and adapted it to my needs. Tutorial notebook can be found here https://github.com/tensorflow/models/blob/r1.13.0/research/object_detection/object_detection_tutorial.ipynb.

  3. Inference frames individually
      - The program has two sections for inference; a section for pure TensorFlow and frozen model, and another section for Inference Engine and optimized model.
      - As explained in step 2, each section has two parts: inference frames with people and inference with frames w/out people. 
      - Each section inferences and captures scores of detected objects (if any) for individual frames. Going through all scores, count is kept for true detection and false/no detection based on the confidence (threshold) set by user. In this case threshold is set to .20 by default. The scoring for each frame for both sections is stored under resources/results: for TensorFlow "no-people-results-tf.csv" and "no-people-results-tf", and for OpenVINO Inference Engine "people-results-openvino-cpu.csv" and "no-people-results-openvino-cpu.csv" respectively. Ideally we would have 1050 detections of people in frames and 344 frames without detection, but that of course is not the case as shown in result files.
      - Using the following formula is how the model's accuracy can be calculated: (true detections of frames w/people + true detections of frames wout/people) / total frames. 
      - With time module the duration of inference was captured, these times were aggregated and averaged after all inference completed. Also, the frames per second were calculated for each section; TensorFlow using numpy arrays and Inference Engine using OpenCV.
  
  4. Gather and compare results between pure Tensorflow and Inference Engine
      - The difference between model accuracy pre-conversion and post-conversion was slightly different. The original frozen model (SSDLite MobileNetV2 COCO, pre-conversion) was slightly more accurate than the optimized model (post-conversion) with Model Optimizer (MO).
      - The frozen model (.pb file) was larger in size before conversion to Intermediate Representation (IR) files with MO.
      - The average inference time was much faster with OpenVINO Inference Engine than with pure TensorFlow. In addition, it can also inference with other devices (not just CPU) like the Neural Compute Stick (Myriad VPU). 

      Metric                TensorFlow  OpenVINO  %Difference 
      Inference Time (s):   35.42       13.98     60.52 
      Avg. Inference (ms):  12.70       4.84      61.91 
      Accuracy (%):         92.04       88.95     3.35 
      FPS:                  4.90        99.69     1933.31 
      Model File Size (MB): 18.99       17.19     9.46

To run the Jupyter notebook it is necessary to source the environment variables for OpenVINO prior starting the Jupyter notebook, otherwise there will be an exception thrown by the interpreter when reaching the OpenVINO Inference Engine section. Steps to run the file as follows:
  - Open Linux terminal and source OpenVINO environment variables: $ source /opt/intel/openvino/bin/setupvars.sh
  - Start notebook: $ jupyter notebook
  - Run object_detection-tf-openvino.ipynb within webapp in browser

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

  1. Retail: one use in retail environment could be a store to keep track of how many people visit an aisle (and/or spend certain amount of time), and therefore target certain items that could boost sales from strategic locations.

  2. Indoor Safety: keep track of how many people are indoors and therefore adjust the rate of flow inside a public space (stores, public transportations, etc.). This way public places can ensure the safety of visitors due to sickness concerns.

  3. Public Lighting: efficiently switch public lighting according to people using certain sidewalks, perhaps limit or adjust brightness depending on how many people are using it. It could also be used in Retail stores, turning off lights in certain store areas if no people are nearby.



## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed model. The potential effects of each of these are as follows: the user must account for the environment where the application will be deployed as unaccounted conditions may affect model performance. The recommendation would be to train a model with image data of similar environments to that of the target environment, and/or set expectations for model prior to deployment (i.e. application only needs pedestrian detections during sunny days, therefore if model detects poorly in other scenarios its ok)

As previously stated, lighting conditions could impact the model's ability to recognize objects, as it can alter objects in those frames lighter/darker. For example, a model that has been trained to detect pedestrians on a sunny day with traffic may require different training data from a model that detects pedestrians walking sideways during the night. So it is important to consider lightning conditions and other possible variables that could impact a model's ability to perform detection.

Another thing to consider is the hardware ability and limitations to perform the task of each individual use case. A high precision model that requires high accuracy may require more compute power and compute time, so the edge device must be able to keep up with the requirements of the given use case. Otherwise among other things, there could be latency penalties as the edge device may take longer processing requests for inference and output results. So evaluating and testing the edge device for a given use case is required.


## Model Research

Details on how to convert pre-trained model successfully.

- SSD Lite MobileNet V2 COCO: 
  - Model Source: http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
  - To convert the model to an Intermediate Representation, use the following arguments: 
    --input_model=frozen_inference_graph.pb \
    --input_shape=[1,300,300,3] \
    --reverse_input_channels \
    --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json \
    --tensorflow_object_detection_api_pipeline_config pipeline.config
  - Full Model Optimizer command for Linux: $ python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model=frozen_inference_graph.pb --input_shape=[1,300,300,3] --reverse_input_channels --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config
