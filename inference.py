#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore, IEPlugin

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None 
        self.exec_network = None
        self.infer_request = None
        
    def load_model(self, model, device, input_size, output_size, num_requests, extensions):
        ### TODO: Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        log.info("Create Inference Engine...")
        self.plugin = IECore()
        
        
        ### TODO: Add any necessary extensions ###
        if extensions and 'CPU' in device: 
            self.plugin.add_extension(extensions, device)

        # Read IR as IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        ### TODO: Check for supported layers ###
        supported_layers = self.plugin.query_network(self.network, device)
        not_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("These layers are not supported by the plugin for device {}:\n {}".format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in demo's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
        
        ### TODO: Return the loaded inference plugin ###
        log.info("Reading IR...")
        self.exec_network = self.plugin.load_network(network=self.network, device_name=device, num_requests=0)
  
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))        

        ### Note: You may need to update the function parameters. ###
        return self.plugin, self.get_input_shape()


    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, frame):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        infer_request_handle = self.exec_network.start_async(request_id=0, inputs={self.input_blob: frame})
        
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        status = self.exec_network.requests[0].wait()
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self):
        ### TODO: Extract and return the output results
        output = self.exec_network.requests[0].outputs[self.output_blob]
        ### Note: You may need to update the function parameters. ###
        return output