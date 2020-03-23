print('container start')
try:
  import unzip_requirements
except ImportError:
  pass
print('unzip end')

# gcp
import numpy
import tensorflow
import json

import requests
from io import BytesIO
from werkzeug.exceptions import BadRequest

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from PIL import Image
#from keras.preprocessing import image

import boto3, os, tempfile

print('import end')

# for image file

TEMP_DIR = '/tmp' 
UPLOAD_BUCKET_NAME = os.environ['UPLOAD_BUCKET_NAME']

s3 = boto3.resource('s3')

print('loading None model...')
model = None

print('None model loaded\n')


# global variables, do I need to put these variables in serverless.yml?

CLASS = 'class'
CODE = 'code'
MESSAGE = 'message'
DATA = 'data'


# We keep model as global variable so we don't have to reload it in case of warm invocations

model = None


# basic model architecture

class CustomModel(Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


# target data URL > S3 URL > download from S3 directly
# 이건 왜 있는 거지? 
# 그냥 url 받는 부분 
def fmnist_classifier(event, context):


    body = {}

    params = event['queryStringParameters']
    if params is not None and 'imageKey' in params:
        image_key = params['imageKey']

        print('image key loaded : ', image_key)

        # download image
        print('Downloading image...')
        tmp_image_file = tempfile.NamedTemporaryFile(dir=TEMP_DIR)

        print('tmp_image_file loaded', tmp_image_file)

        #img_object = s3.Bucket(UPLOAD_BUCKET_NAME).Object(image_key)
        img_object = s3.Bucket('image-upload-mokpolar').Object('test.png')

        print('img_object setted', img_object)
        

        img_object.download_fileobj(tmp_image_file)
        print('Image downloaded to', tmp_image_file.name)

        #  load and preprocess the image
        img = (numpy.array(Image.open(tmp_image_file))/255)[numpy.newaxis,:,:,numpy.newaxis] # > 이 부분은 fashion mnist size에 맞게 변경 28, 28인듯 255로 나눠줘야 함

        # img는 닫아 줄 필요가 없나.. 일단 돌려보자. 

        # 여기서 img 만 잘 들어가면 예측이 될 것이다. 
        predicted_class = predict_mnist_internal(img)

        data = {
            CLASS: predicted_class,
        }
        

        body['message'] = 'OK'
        body['predictions'] = data

    response = {
        "statusCode": 200,
        "body": json.dumps(body),
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Content-Type": "application/json"
        }
    }


def predict_mnist_internal(img):

    global model
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Model load which only happens during cold starts
    if model is None:
        model = CustomModel()
        model.load_weights('fashion_mnist_weights')


    predictions = model.call(img)
    print(predictions)
    predicted_class = class_names[numpy.argmax(predictions)]
    print("Image is " + predicted_class)
    return predicted_class
