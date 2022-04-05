# Age and Gender Estimation with AWS
 This repository is ment to help others implement and understand the pipline to estimate age and gender of a person based on one facial image
 
## Introduction
During the final semester of my Master’s degree, I took a course to gain experience in cloud computing. 
I learned how to display, render, deploy and interpret it in the context of a real or simulated close-up loop type cloud based engineering system. 
Furthermore, I acquired a knowledge of how to communicate the shortcomings and vulnerabilities of such systems, including plug-and-play systems using pre-trained off-the-shelf deep learning models, when integrated into a decision-making system. 

## Architecture
Figure 1 displays the cloud architecture used during implementation. 
It can be observed that the user is uploading an image to the cloud via an API-Gateway. 
Therefore, the image is being decoded as a json body on the end-device. 
This upload automatically triggers the creation of a Lambda-Instance which is going to perform the computation. 
After the Lambda-Function successfully determines the age and gender, it will write the results in the original image and send it back to the user via the API. 
Finally, the end-device will encode the received data and display the original image with the results of the age and gender estimation. 

### Lambda Functions
In the folder Lambda Function 

![Figure 1: Cloud Architecture for Implementation](https://github.com/[Marcus-Koenig]/[Age-and-Gender-Estimation-with-AWS]/blob/[main]/Architecture.png?raw=true)
