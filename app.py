from flask import Flask, render_template, request
import pickle
import numpy as np

# 우리 모델 불러오기
# model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)

@app.route("/ping", methods=['GET'])
def ping():
    return "통신 테스트"

@app.route('/')
def main():
    return render_template('index.html')

# @app.route('/', methods=['POST'])
# def home():
#     data1 = request.form['a']
#     data2 = request.form['b']
#     data3 = request.form['c']
#     data4 = request.form['d']
#     arr = np.array([[data1, data2, data3, data4]])
#     pred = model.predict(arr)
#     return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)