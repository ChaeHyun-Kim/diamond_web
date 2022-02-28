from flask import Flask, render_template, send_from_directory
import tensorflow as tf
import numpy as np



app = Flask(__name__)

def predict():
    model = tf.keras.models.load_model('model/lstm_best_model.h5')
    return model.layers

@app.route('/', methods=['GET'])
def main():

    return render_template('index.html')


@app.route('/final', methods=['GET'])
def home():
    value = predict()
    return render_template('result.html', memo=value)


@app.route('/lib/<path:path>')
def send_js(path):
    return send_from_directory('lib', path)

@app.route("/ping", methods=['GET'])
def ping():
    return "통신 테스트"


if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')
