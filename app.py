import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import pickle


app = Flask(__name__)
def loadBinary():
    #load binary model
    file = open('model_rf_2', 'rb')
    binary_model = pickle.load(file)
    return binary_model

def loadVGG():
    #load binary model
    vgg_model=load_model("vgg_final.h5")
    return vgg_model

def loadMulti():
    #load binary model
    file = open('final_model.pkl', 'rb')
    multilabel_model=pickle.load(file)
    return multilabel_model

def binary_predict(img_path, model,vgg_model):
    binary_model=model
    vgg=vgg_model
    img = image.load_img(img_path, target_size=(400, 400))
    x = image.img_to_array(img)
    X = np.expand_dims(x, axis=0)
    preds = vgg.predict(X)
    val = preds.reshape(preds.shape[0], -1)
    output=binary_model.predict(val)
    final=output[0]
    return final


def multilabel_predict(img_path, multilabel_model,vgg_model):
    multimodel=multilabel_model
    vgg=vgg_model
    img = image.load_img(img_path, target_size=(400, 400))
    x = image.img_to_array(img)
    X = np.expand_dims(x, axis=0)
    preds = vgg.predict(X)
    val = preds.reshape(preds.shape[0], -1)
    diseases=multimodel.predict(val)
    return diseases

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/i', methods=['GET'])
def i():
    # Main page
    return render_template('index.html')

@app.route('/p', methods=['GET'])
def p():
    # Main page
    return render_template('predict.html')

@app.route('/c', methods=['GET'])
def c():
    # Main page
    return render_template('contactus.html')



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
             basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        binaryModel = loadBinary()
        vggModel = loadVGG()
        multilabelModel = loadMulti()
        binary_preds = binary_predict(file_path, binaryModel,vggModel)
        # print(binary_preds)
        #binary_preds=1
        if(binary_preds==1):
            multi_preds = multilabel_predict(file_path, multilabelModel,vggModel)
            print(type(multi_preds))

            values=multi_preds.tolist()
            values=values[0]
            diseases=[]
            classes = ['DR', 'MYA', 'ODC', 'CRVO', 'AH','AION','MHL']
            for i in range(len(classes)):
                if (values[i] == 1):
                    diseases.append(classes[i])
            binary_result="Disease detected are : "
            listToStr = ' '.join([str(elem) for elem in diseases])
            result=binary_result+listToStr
            return render_template('predict.html',
                                   prediction_text='Diagnosis is {} '.format(result))
        binary_result = "No disease detected"

        return render_template('predict.html',
                               prediction_text='Diagnosis is {} '.format(binary_result))
    return None

if __name__ == '__main__':
    app.run(debug=True)
