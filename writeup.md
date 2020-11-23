# Project Write-Up

For this project I decided to use ssdlite_mobilenet_v2_coco_2018_05_09 model

## Explaining Custom Layers

The process behind converting custom layers involves implementing functions to extend the Model Optimizer as well as the Inference Engine.

Some of the potential reasons for handling custom layers are to add support for layers not supported by the toolkit for a framework (i.e. TF, Caffe, ONNX, etc.) or target device (i.e. CPU, GPU, VPU, etc.) of choice.
It could be that your custom trained model may have layers that are all supported by the CPU, but some of these layers may not work on GPU/VPU. In this case if you want your model to run on GPU/VPU, as a user you must implement these custom layers to extend this functionality to the Model Optimizer and Inference Engine. 

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

1. Retail: one use in retail environment could be a store to keep track of how many people visit an aisle (and/or spend certain amount of time), and therefore target certain items that could boost sales from strategic locations.

2. Indoor Safety: keep track of how many people are indoors and therefore adjust the rate of flow inside a public space (stores, public transportations, etc.). This way public places can ensure the safety of visitors due to sickness concerns.

3. Public Lighting: efficiently switch public lighting according to people using certain sidewalks, perhaps limit or adjust brightness depending on how many people are using it. It could also be used in Retail stores, turning off lights in certain store areas if no people are nearby.


## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed model. The potential effects of each of these are as follows: the user must account for the environment where the application will be deployed as unaccounted conditions may affect model performance. The recommendation would be to train a model with image data of similar environments to that of the target environment, and/or set expectations for model prior to deployment (i.e. application only needs pedestrian detections during sunny days, therefore if model detects poorly in other scenarios its ok)

As previously stated, lighting conditions could impact the model's ability to recognize objects, as it can alter objects in those frames lighter/darker. 
For example a model that has been trained to detect pedestrians on a sunny day with traffic may require different training data from a model that detects pedestrians walking sideways during the night. So it is important to consider lightning contitions and other possible variables that could impact a model's ability to perform detection.


## Model Research

Details on how to convert pre-trained model successfully.

- SSD Lite MobileNet V2 COCO: 
  - Model Source: http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments: 
    --input_model=frozen_inference_graph.pb \
    --input_shape=[1,300,300,3] \
    --reverse_input_channels \
    --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json \
    --tensorflow_object_detection_api_pipeline_config pipeline.config
  - Model Optimizer command for Linux: $ python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model=frozen_inference_graph.pb --input_shape=[1,300,300,3] --reverse_input_channels --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config 
  - Model Optimizer command for Windows: $ python "C:\Program Files (x86)\Intel\openvino\deployment_tools\model_optimizer\mo_tf.py" --input_model=frozen_inference_graph.pb  --input_shape=[1,300,300,3] --reverse_input_channels --transformations_config "C:\Program Files (x86)\Intel\openvino\deployment_tools\model_optimizer\extensions\front\tf\ssd_v2_support.json" --tensorflow_object_detection_api_pipeline_config pipeline.config