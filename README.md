# Age and Gender Estimation with AWS
 This repository is ment to help others implement and understand the pipline to estimate age and gender of a person based on one facial image
 
## Introduction
During the final semester of my Master’s degree, I took a course to gain experience in cloud computing. 
I learned how to display, render, deploy and interpret it in the context of a real or simulated close-up loop type cloud based engineering system. 
Furthermore, I acquired a knowledge of how to communicate the shortcomings and vulnerabilities of such systems, including plug-and-play systems using pre-trained off-the-shelf deep learning models, when integrated into a decision-making system. 

This project is about an age and gender estimation based on one facial image. 
The pipeline is based on Amazon Web Services (AWS) and an interactive user interface. 

## Architecture
Figure 1 displays the cloud architecture used during implementation. 
It can be observed that the user is uploading an image to the cloud via an API-Gateway. 
Therefore, the image is being decoded as a json body on the end-device. 
This upload automatically triggers the creation of a Lambda-Instance which is going to perform the computation. 
After the Lambda-Function successfully determines the age and gender, it will write the results in the original image and send it back to the user via the API. 
Finally, the end-device will encode the received data and display the original image with the results of the age and gender estimation. 

### Lambda Functions
In the folder [Lambda Function](/Lambda-Function) are two code examples given. 
[Dropdown_Button.py](/Lambda-Function/Dropdown_Button.py) is used in the user interface to get the example files from the S3 Bucket. 
[Inferencing.py](/Lambda-Function/Inferencing.py) contains the python code to inference the uploaded image from the user. 
Note that [Inferencing.py](/Lambda-Function/Inferencing.py) requires additional libraries which can be included via Lambda Layers. 

![Cloud Architecture](/Architecture.png?raw=true)
*Figure 1: Cloud Architecture for Implementation*


## Model(s)
The original models were taken from [https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/). 
Those models were pre-trained on the IMDb-WIKI data set.
The decision to use pre-trained models is based on the limited time-frame of the project and the focus on the cloud implementation. 
To enable both models, one for age and one for gender estimation, to be used in a Lambda-Instance, they have to be modified. 
First, both models have to be transformed from the original _caffe_ structure to _onnx_. 
Details of that transformation can be found in [https://pypi.org/project/caffe2onnx/](https://pypi.org/project/caffe2onnx/). 
After the transformation, both models are going to have over 500 MByte each. 
To reduce their size, they have to be quantized to _uint8_ format. 
This reduces the required memory for the Lambda-Instance and increases the inference speed. 

## User Interface
To create a user interface which enables edge-devices to use this project without additional computational resources on their own device, a ‘React-App’ was used. 
This application can be accessed via any web browser. 
The user can choose to upload one image from his device or choose one of the provided example images. 
These example images are located in a AWS S3 Bucket and can be changed/adjusted independently from the web application directly on the cloud. 

The code to generate the user interface and connect it with the APIs is provided under [User-Interface](/User-Interface). 
