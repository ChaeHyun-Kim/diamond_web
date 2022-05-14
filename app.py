# -*- coding: utf-8 -*-
from flask import Flask, jsonify, render_template, send_from_directory,request,redirect,url_for
import tensorflow as tf
import numpy as np

from model.final_model import *

from flask import request

app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def main():
    return render_template('index.html')


@app.route("/final", methods=["POST", "GET"])
def final():
    jsonData=request.get_json()
    print("입력데이터: ",jsonData)
    new_sentence = jsonData["ad"] 
    result = ad_predict(new_sentence)
    
    return jsonify({"sentence": new_sentence, "predict_result":result})

@app.route("/result_render", methods=["POST", "GET"])
def result_render():
    return render_template('result.html')
    
@app.route("/restart", methods=["POST", "GET"])
def restart():
    return render_template('index.html')

@app.route('/team_info', methods=["POST", "GET"])
def team_info():
    return render_template('team_info.html')


@app.route('/topic_info', methods=["POST", "GET"])
def topic_info():
    return render_template('topic_info.html')


@app.route('/lib/<path:path>')
def send_js(path):
    return send_from_directory('lib', path)

@app.route("/ping", methods=['GET'])
def ping():
    return "통신 테스트"

if __name__ == "__main__":
    # ad_predict("바르기만 해도 살이 빠져요. 혈중 중성지방 수치를 낮추기 위해선 탄수화물의 섭취량을 제한하고 총 지방과 단백질을 적정하게 섭,취하여 표준체중을 유지하여야합니다. 따라서 기름이많은 부위의 고기와식용유, 버터가 많이 들어간,케이크, 머핀의 섭취를주의해야하며, 탄수회물을 섭취할 때에도 쌀밥대신 잡곡밥으로 섭취하는 것이,좋습니다. 술은 칼로리가 높으며 결들여 먹는 안주는 기름진 음식이 많기 때문에 가급적 마시지 않는 것,이 좋습니다. 중성지방을 원천적으로 줄이기 위해선 식품의 요리방법도 중요한데, 예를 들어 재료 튀기,거나 볶는 방법 보다는 찌거나 삶는 방법을 이용하는 것이 좋습니다.비타민 C, 비타민 E : 유해 산소로부터 세포를 보호하는 대표적인 성분입니다. 또한 비타민 C는 결합조직의 형성과 기능 유지에 필요한 성분입니다.")
    
    app.run(debug=True, port=5000, host='0.0.0.0')
    