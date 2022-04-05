import sys
import os
import numpy as np
from datetime import datetime
import boto3
import onnxruntime as rt
import cv2
import base64
import json


def preprocess(img_data):
    margin = 0.4
    shape_frame = img_data.shape
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
    face_rects = detector.detectMultiScale(gray, 1.3, 5)
    N = len(face_rects)
    inputs = np.zeros((N, 3, 224, 224), dtype=np.float32)
    i = 0
    for (x, y, w, h) in face_rects:
        # check if y window is bigger than the frame
        if y - h * margin / 2 <= 0:
            y_low = 0
        elif y + h * (1 + margin) >= shape_frame[0]:
            y_low = shape_frame[0] - h * (1 + margin)
        else:
            y_low = y - h * margin / 2
        y_low = int(y_low)

        # check if x window is bigger than the frame
        if x - w * margin / 2 <= 0:
            x_low = 0
        elif x + w * (1 + margin) >= shape_frame[1]:
            x_low = shape_frame[1] - w * (1 + margin)
        else:
            x_low = x - w * margin / 2
        x_low = int(x_low)

        # select the face reagin from the frame
        y_high = int(y_low + h * (1 + margin))
        x_high = int(x_low + w * (1 + margin))
        
        face = img_data[y:y + h, x:x + w, :]
        inputs[i, :, :, :] = (np.transpose(cv2.resize(face, (224, 224)), (2, 0, 1)))
        i += 1
    return inputs, face_rects


def makeInference_age(sess, inputs):
    N = inputs.shape[0]
    age = []
    for i in range(N):
      input = inputs[i,:,:,:]
      input = np.expand_dims(input, 0)
      pred_onx = sess.run(None, input_feed={'input': input})
      age_current = round(sum(pred_onx[0][0] * list(range(0, 101))), 1)
      #age_current = np.argmax(pred_onx)
      age.append(age_current)
      #print(age_current)
    #scores = softmax(pred_onx)
    return age

def makeInference_gender(sess, inputs):
    N = inputs.shape[0]
    gender = []
    for i in range(N):
      input = inputs[i,:,:,:]
      input = np.expand_dims(input, 0)
      pred_onx = sess.run(None, input_feed={'input': input})
      gender.append(pred_onx)
    #scores = softmax(pred_onx)
    return gender


def postprocess(img_data, face_rects, inferences_age, inferences_gender, Gender_dict):
    i = 0
    num_faces = len(face_rects)
    shape_frame = img_data.shape
    male_sign = cv2.imread(os.path.join(lambda_tmp_directory, male_sign_name))
    female_sign = cv2.imread(os.path.join(lambda_tmp_directory, female_sign_name))
    for (x, y, w, h) in face_rects:
        inference_gender = inferences_gender[i]
        gender = Gender_dict[int(np.argmax(inference_gender[:2]))]
        age = int(inferences_age[i])
        if gender == 'male':
            color = (235, 216, 173)
            gender_sign = male_sign
            cv2.putText(gender_sign, '{}'.format(age), (12, 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
        else:
            color = (102, 0, 204)
            gender_sign = female_sign
            cv2.putText(gender_sign, '{}'.format(age), (15, 42), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 0), 2)
        frame_thickness = 4
        cv2.rectangle(img_data, (x, y), (x + w, y + h), color, frame_thickness)
        width = int(w / 3.5)
        if width / shape_frame[1] < 1 / 30: width = shape_frame[1] // 30
        dims = (width, int(gender_sign.shape[0] * (width / gender_sign.shape[1])))
        gender_sign = cv2.resize(gender_sign, dims)
        for c in range(gender_sign.shape[2]):
            img_data[y + h - gender_sign.shape[0]:y + h, x + w - gender_sign.shape[1]:x + w, c][gender_sign[:, :, c] < 10] = color[c]

    #cv2.putText(img_data, 'Gender: {}, Age: {}'.format(gender, age), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5,
        #            (255, 255, 255))
        #print('Gender: {}, Age: {}'.format(gender, age))
        i += 1
    return True


def lambda_handler(event, context):
    # s3_bucket_name = "age-gender-estimation-marcusko"
    global lambda_tmp_directory
    lambda_tmp_directory = "/tmp"
    model_age_file_name = "model_age.quant.onnx"
    model_gender_file_name = "model_gender.quant.onnx"
    GENDER_DICT = ['female', 'male']
    s3_bucket_name_output = "age-gender-estimation-results-marcusko"
    s3_bucket_name_input = "age-gender-estimation-marcusko"
    global male_sign_name
    global female_sign_name
    male_sign_name = "male_sign.png"
    female_sign_name = "female_sign.png"
    input_file_name = "input.png"
    output_file_name = "result.png"


    try:
        # Download test image and model from S3.
        client = boto3.client('s3')
        client.download_file(s3_bucket_name_input, model_age_file_name, os.path.join(lambda_tmp_directory, model_age_file_name))
        client.download_file(s3_bucket_name_input, model_gender_file_name, os.path.join(lambda_tmp_directory, model_gender_file_name))
        client.download_file(s3_bucket_name_input, male_sign_name, os.path.join(lambda_tmp_directory, male_sign_name))
        client.download_file(s3_bucket_name_input, female_sign_name, os.path.join(lambda_tmp_directory, female_sign_name))
    except Exception as error:
        return {
            "statusCode": 400,
            "errorMessage": json.dumps(
                {
                    "outputResultsData": str(error)
                }
            )
        }

    try:
        # Extracting and saving image from POST request body.
        imageDataB64 = event["image"]
        imageBytes = base64.b64decode(imageDataB64.encode("utf8"))
        with open(os.path.join(lambda_tmp_directory, input_file_name), "wb") as f:
            f.write(imageBytes)
            f.close()
    except Exception as error:
        return {
            "statusCode": 400,
            "errorMessage": json.dumps(
                {
                    "outputResultsData": str(error)
                }
            )
        }

    # Import input image and preprocess it.
    # image = Image.open("/content/drive/MyDrive/EECS605/Project/Age_estimation/test.jpg").convert('RGB')
    # image = np.asarray(image)
    #print(os.path.join(lambda_tmp_directory, input_file_name), input_file_name, s3_bucket_name_input)
    #print(os.listdir(lambda_tmp_directory))

    try:
        image = cv2.imread(os.path.join(lambda_tmp_directory, input_file_name))
        processed_image, facial_rectangels = preprocess(image)
    except Exception as error:
        return {
            "statusCode": 400,
            "errorMessage": json.dumps(
                {
                    "outputResultsData": str(error)
                }
            )
        }

    # Make inference using the ONNX model.
    sess_age = rt.InferenceSession(os.path.join(lambda_tmp_directory, model_age_file_name))
    sess_gender = rt.InferenceSession(os.path.join(lambda_tmp_directory, model_gender_file_name))
    inferences_age = makeInference_age(sess_age, processed_image)
    inferences_gender = makeInference_gender(sess_gender, processed_image)

    # Postprocessing to write the gender and age in the image
    postprocess(image, facial_rectangels, inferences_age, inferences_gender, GENDER_DICT)

    # image.save("/content/drive/MyDrive/EECS605/Project/Age_estimation/test_result.png")
    # cv2.imshow(image)
    cv2.imwrite(os.path.join(lambda_tmp_directory, output_file_name), image)

    # Get today's date and append to the filename.
    current_date_time = str(datetime.now())

    # Convert output file into bytes.
    with open(os.path.join(lambda_tmp_directory, output_file_name), "rb") as outputFile:
        outputFileBytes = outputFile.read()
    outputFileBytesB64 = base64.b64encode(outputFileBytes).decode("utf8")

    # Send it back as a response.
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "outputResultsData": outputFileBytesB64,
                "fileType": ".png"
            }
        )
    }

# Uncomment to run locally.
# lambda_handler(None, None)
