from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
from io import BytesIO
import json
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np


app = Flask(__name__)
UPLOAD_FOLDER = '/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/home', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/test', methods=['GET'])
def test():
    return render_template('uploader/index.html')


@app.route('/upload', methods=['GET'])
def upload():
    return render_template('upload.html')


@app.route("/data", methods=["POST"])
def get_data():
    # receive the JSON data from the request
    data = request.get_json()
    print("hi:", data['result'])
    ans = data['result']
    accuracy = {"very good": ">95", "good": "85-95",
                "bad": "85-75", "very bad": "<75"}
    print(accuracy[ans])
    # do some processing with the data

    # render the template and pass the processed data as a parameter
    return render_template('esha_card/pg2.html', result=ans, accuracy=accuracy[ans])


@app.route('/uploadfile', methods=['POST'])
def uploadfile():
    if request.method == 'POST':
        file = request.files['file']
        file.save(os.path.join(app.root_path, f"uploads/{file.filename}"))
        img_path = os.path.join(app.root_path, f"uploads/{file.filename}")
        print(img_path)

        # return ""
        # return redirect(url_for('esha_card/pg2.htmldex'))
        # render_template('esha_card/pg2.html',result = result,img_path=img_path)

        model = load_model('model2_resnet.h5')

        img = image.load_img(img_path, target_size=(224, 224))

        imgResult = img_to_array(img)
        imgResult = np.expand_dims(imgResult, axis=0)
        imgResult = imgResult / 255.

        preds = model.predict(imgResult)

        # create a list containing the class labels
        class_labels = ['bad', 'good', 'very bad', 'very good']

        # find the index of the class with maximum score
        pred = np.argmax(preds, axis=-1)
        # print the label of the class with maximum score
        print(class_labels[pred[0]])
        # return class_labels[pred[0]]
        data = {"result": class_labels[pred[0]]}

        response = app.test_client().post(
            "/data", json=data)

        # check the response status code
        assert response.status_code == 200

        # return the response data
        return response.data


@app.route('/receivedata', methods=['POST'])
def receive():
    print(request.form['myData'])
    return ''


@app.route('/cards', methods=['GET'])
def cards():
    return render_template('esha_card/pg2.html', result="Hello", img_path="/Users/droom/Documents/newunscript/uploads/Very Bad Images[Accuracy_75]260.png")


@app.route("/example", methods=["POST"])
def example():
    print("helo")
    data = request.get_json()
    print(data['img'])

    # un comment the below for prediction using the model

    """ model = load_model('model2_resnet.h5')
    img_path = data["img"]
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    imgResult = img_to_array(img)
    imgResult = np.expand_dims(imgResult, axis=0)
    imgResult = imgResult / 255.

    preds = model.predict(imgResult)

    # create a list containing the class labels
    class_labels = ['bad', 'good', 'very bad', 'very good']

    # find the index of the class with maximum score
    pred = np.argmax(preds, axis=-1)
    # print the label of the class with maximum score
    print(class_labels[pred[0]])

    return class_labels[pred[0]] """
    return ""


if __name__ == '__main__':
    app.run(debug=True)
