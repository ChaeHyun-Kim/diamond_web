from flask import Flask, render_template,request,redirect,url_for
#  request
# import pickle
# import numpy as np
import tensorflow as tf
# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers.convolutional import Conv2D
# from keras.layers.convolutional import MaxPooling2D
# from keras.preprocessing.image import ImageDataGenerator

# dir_path ='C://Users//kjh00//OneDrive//바탕 화면//크리마란- 산학연계//'
# MODEL_NAME = 'lstm_best_model.h5'


# model = keras.models.load_model(dir_path+MODEL_NAME)
# model.summary()


app = Flask(__name__)

@app.route("/ping", methods=['GET'])
def ping():
    return "통신 테스트"

@app.route('/', methods=['GET','POST'])
def main():
    return render_template('index.html')

@app.route('/method', methods=['GET','POST'])
def method():
    if request.method=='GET':
        val1=request.args['memo']
        dir_path = 'C:/Users/com/diamond_web/lstm_best_XAImodel.h5' #모델 불러오기
        loaded_model = tf.keras.models.load_model(dir_path)


        return redirect(url_for('result',val=val1))

@app.route('/<val>')
def result(val):
    print(val)
    #value1 = 'hello, world123'
    print("slsls")
    return render_template('result.html',memo1=val)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
# from flask import Flask 
# app = Flask(__name__) 
# @app.route('/') 
# def home(): 
#     return 'Hello World' 
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000)
