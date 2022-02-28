
from flask import Flask, render_template, send_from_directory,request,redirect,url_for
import tensorflow as tf
import numpy as np



app = Flask(__name__)
# 채현시작
# 모델 실행 함수
def predict():
    model = tf.keras.models.load_model('model/lstm_best_model.h5')
    return model.layers

@app.route('/', methods=['GET','POST'])
def main():
    return render_template('index.html')

@app.route('/final', methods=['GET'])
def home():
    value = predict()
    return render_template('result.html', memo=value)

#   
  
  
# 주현시작
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

# 주현끝

@app.route('/lib/<path:path>')
def send_js(path):
    return send_from_directory('lib', path)

@app.route("/ping", methods=['GET'])
def ping():
    return "통신 테스트"


if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')
