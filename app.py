from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

dir_path ='C://Users//kjh00//OneDrive//바탕 화면//크리마란- 산학연계//'
MODEL_NAME = 'lstm_best_model.h5'


model = keras.models.load_model(dir_path+MODEL_NAME)

app = Flask(__name__)
@app.route('/')
def main():

    return render_template('index.html')

@app.route("/calculate",  methods=['POST','GET'])
def index():
    if request.method == 'POST':
        # temp = request.form['num']
        pass
    elif request.method == 'GET':
        temp = request.args.get('memo')
        ## 넘겨받은 문자
        print(temp)
        return render_template('index.html', memo=temp)
    ## else 로 하지 않은 것은 POST, GET 이외에 다른 method로 넘어왔을 때를 구분하기 위함


@app.route('/', methods=['POST'])
def home():
    value = 'hello, world'
    return render_template('index.html', memo=value)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)