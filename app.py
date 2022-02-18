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

# dir_path ='C://Users//kjh00//OneDrive//바탕 화면//크리마란- 산학연계//'
# MODEL_NAME = 'lstm_best_model.h5'


# model = keras.models.load_model(dir_path+MODEL_NAME)
# model.summary()


app = Flask(__name__)

@app.route("/ping", methods=['GET'])
def ping():
    return "통신 테스트"

@app.route('/')
def main():
    value = 'hello, world'

    return render_template('index.html', memo=value)


@app.route('/', methods=['POST'])
def home():
    value = 'hello, world'
    return render_template('index.html', memo=value)


if __name__ == "__main__":
    app.run(debug=True)