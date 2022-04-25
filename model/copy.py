# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from konlpy.tag import Okt
import tensorflow as tf
from hanspell import spell_checker
import kss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


path = 'model/NanumGothic.ttf'
best_font = fm.FontProperties(fname=path, size=50).get_name()
plt.rc('font', family='NanumGothic')


class diamond: 
  def __init__(self, ad):
    self.model_load()
    self.ad = ad
    self.li = []

    #맞춤법+전처리
    spelled_sent = spell_checker.check(self.ad)
    self.ad = spelled_sent.checked
    self.ad = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', self.ad)
    for sent in kss.split_sentences(self.ad):
      self.li.append(sent)
    self.ad_df = pd.DataFrame({"광고내용":self.li})
    self.kopro_df = self.ko_processing() 
    self.pro_df = self.processing()

    self.pro_df['score'] = np.nan
    self.pro_df["score"] = self.pro_df["score"].astype(float)
    self.pro_df.rename(columns = {'광고내용':'문장'}, inplace = True)

    self.input = self.tokenizing1(self.pro_df)

    #토큰화

    print("광고 내 문장")
    num = 1
    for sen in self.pro_df['문장']:
      print("문장"+str(num)+" : " + sen)
      num += 1

    if len(self.pro_df)>1:
      self.ad2_predict()

    else:
      self.ad1_predict()

   


  def ad1_predict(self):
    #문장 분리

    #문장이 1개
    print("\n\n문장 별 예측 결과")
    self.make_plot(self.pro_df, self.input,0, 0.5)
    self.pro_df = self.pro_df.reset_index(drop=True)
    if(self.pro_df.loc[0,'score'] > 0.5):
      print("\n최종 예측 결과 : 해당 광고는 {:.2f}% 확률로 허용광고입니다.\n".format(self.pro_df['score'][0] * 100))

    else:
      self.pro_df['score'] = 1-self.pro_df['score']
      print("\n최종 예측 결과 : 다음 문장 때문에 해당 광고는 {:.2f}% 확률로 허위광고입니다.\n".format((self.pro_df['score'][0]) * 100))
      self.pro_df.rename(columns = {'문장':'위험 문장'}, inplace = True)
      self.pro_df.rename(columns = {'score':'위험도'}, inplace = True)
      # display(ad_df.loc[:,['위험 문장', '위험도']])
      return self.pro_df.loc[:,['위험 문장', '위험도']]


  def ad2_predict(self):
      #문장이 2개 이상

    #모델 예측
    print("\n\n문장 별 예측 결과")
    for i in range(len(self.pro_df)):
      self.make_plot(self.pro_df, self.input, i, 0.3)



    #score
    danger = self.pro_df.loc[self.pro_df['score']<0.3].copy()
    danger.rename(columns = {'문장':'위험 문장'}, inplace = True)
    danger.rename(columns = {'score':'위험도'}, inplace = True)
    danger = danger.reset_index(drop=True)
    safety = self.pro_df.loc[self.pro_df['score']>=0.3].copy()  

    if len(danger)!=0:
      danger['위험도'] = 1-danger['위험도']
      print("\n최종 예측 결과 : 다음 문장 때문에 해당 광고는 {:.2f}% 확률로 허위광고입니다.\n".format((danger['위험도'].mean()) * 100))
      # display(danger.loc[:,['위험 문장', '위험도']])
      return danger.loc[:,['위험 문장', '위험도']]

    else:
      print("\n최종 예측 결과 : 해당 광고는 {:.2f}% 확률로 허용광고입니다.\n".format(safety['score'].mean() * 100))


  # # 중복 제거

  # def duplicatesRemove(self,data) :
  #   data.drop_duplicates(subset = ['광고내용'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
  #   data['광고내용'] = data['광고내용'].str.replace("[^가-힣 ]","") # 정규 표현식 수행
  #   data['광고내용'] = data['광고내용'].str.replace('^ +', "") # 공백은 empty 값으로 변경
  #   data['광고내용'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경

  #   data = data.dropna(how='any') # Null 값 제거

  #   data.reset_index(drop=True, inplace=True)

  #   print('전처리 후 데이터 수 :',len(data))

  #   return data
  #hanspell 맞춤법 검사

  # def spellcheck(self):
  #   data['hanspell']=''
  #   for i in range(len(data)):
  #     spelled_sent = spell_checker.check(data["광고내용"][i])
  #     data['hanspell'][i] = spelled_sent.checked
  #   data.drop("광고내용", axis = 1, inplace = True)
  #   data = data[['hanspell','label']]
  #   data.rename(columns={'hanspell':'광고내용'}, inplace = True)
  #   return data

  #토큰화+불용어제거
  def ko_processing(self):
    stopwords = ['수','로','로부터','되다','다','든지','께','께서', '이라고','이다','것','의','가','하고','하고는','이','은','이므로','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','을']
    #stopwords = ['되다','인', '줄', '되어다', '것', '이라고','께','께서','든','든지','로','더러','며','만은','조차','처럼','한테','하고','하고는','커녕','나','이다','지요','이므로','있다','이다','의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','을']
    okt = Okt()
    self.ad_df['ko_processing'] = np.nan
    self.ad_df["ko_processing"] = self.ad_df["ko_processing"].astype(object)

    X = []
    for sentence in self.ad_df['광고내용']:
        tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
        stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
        X.append(stopwords_removed_sentence)

    #ko_procssing
    for i in range(len(self.ad_df)):
        self.ad_df['ko_processing'][i]=X[i][:]
    self.ad_df = self.ad_df.dropna(how='any') # Null 값 제거
    return self.ad_df

  #정수 인코딩
  def processing(self):
    self.kopro_df["processing"] = np.nan
    self.kopro_df["processing"] = self.kopro_df["processing"].astype(object)


    X = self.kopro_df['ko_processing'].to_list()
    processing = self.tokenizer.texts_to_sequences(X)

    #procssing
    for i in range(len(self.kopro_df)):
      self.kopro_dfdf['processing'][i]=processing[i][:]
    
    self.kopro_df = self.kopro_df.dropna(how='any') # Null 값 제거
    return self.kopro_df



  def tokenizing(self,df):
    X = []
    for sen in df['processing']:#processing에서 한줄을 가져옴
        word_arr =[]
        for word in sen:#[54,23,14]
            if word in self.word_index:
                word_arr.append(float(self.con[self.con[:,0] ==word,1])) #array([-0.95]) toal_value 가 들어감
            else:
                word_arr.append(0) #없으면 0이 들어감 

        [word_arr.insert(0,0) for i in range(141-len(word_arr))] #패딩하기
        X.append(word_arr)

    X = pd.DataFrame(X)
    X = X.loc[:,:140]
    X= X.to_numpy()
    X = X.reshape(-1,141,1)
    X.shape
    return X

  def model_load(self):
    MODEL_NAME = 'model/lstm_best_XAImodel.h5'


    import pickle
    self.loaded_model = tf.keras.models.load_model(MODEL_NAME)
    with open('model/tokenizer.pickle', 'rb') as handle:
        self.tokenizer = pickle.load(handle)

    self.con = np.load('model/con_save.npy')
    self.word_index = [row[0] for row in self.con]
    self.word_index = list(map(int, self.word_index))

    #names = [weight.name for layer in self.loaded_model.layers for weight in layer.weights]
    self.weights = self.loaded_model.get_weights()

    kernel_weights = self.weights[0]
    recurrent_kernel_weights = self.weights[1]
    bias = self.weights[2]

    self.n = 1
    units = 141  # LSTM layers  

    self.Wi = kernel_weights[:, 0:units]
    self.Wf = kernel_weights[:, units:2 * units]
    self.Wc = kernel_weights[:, 2 * units:3 * units]
    self.Wo = kernel_weights[:, 3 * units:]


    self.Ui = recurrent_kernel_weights[:, 0:units]
    self.Uf = recurrent_kernel_weights[:, units:2 * units]
    self.Uc = recurrent_kernel_weights[:, 2 * units:3 * units]
    self.Uo = recurrent_kernel_weights[:, 3 * units:]


    self.bi = bias[0:units]
    self.bf = bias[units:2 * units]
    self.bc = bias[2 * units:3 * units]
    self.bo = bias[3 * units:]

  def sigmoid(x):
      return 1 / (1 + np.exp(-x))




  def make_plot(self, test_df, test_li, number, standard_score):
      
      ht_1 = np.zeros(self.n * self.units).reshape(self.n, self.units)
      Ct_1 = np.zeros(self.n * self.units).reshape(self.n, self.units)

      h_t_value = []
      influence_h_t_value = []
      #test_li=test_li[0] #이중리스트로 변환

      for t in range(0, len(test_li[number,:])):
          xt = np.array(test_li[number,t])
          ft = self.sigmoid(np.dot(xt, self.Wf) + np.dot(ht_1, self.Uf) + self.bf)  # forget gate

          influence_ft = (np.dot(ht_1, self.Uf))/(np.dot(xt, self.Wf) + np.dot(ht_1, self.Uf) + self.bf) * ft

          it = self.sigmoid(np.dot(xt, self.Wi) + np.dot(ht_1, self.Ui) + self.bi)  # input gate
          influence_it = (np.dot(ht_1, self.Ui))/(np.dot(xt, self.Wi) + np.dot(ht_1, self.Ui) + self.bi) * it

          ot = self.sigmoid(np.dot(xt, self.Wo) + np.dot(ht_1, self.Uo) + self.bo)  # output gate
          influence_ot = np.dot(ht_1, self.Uo) / (np.dot(xt, self.Wo) + np.dot(ht_1, self.Uo) + self.bo) * ot

          gt =  np.tanh(np.dot(xt, self.Wc) + np.dot(ht_1, self.Uc) + self.bc)
          influence_gt =np.dot(ht_1, self.Uc) / (np.dot(xt, self.Wc) + np.dot(ht_1, self.Uc) + self.bc) * gt

          Ct = ft * Ct_1 + it * gt
          influence_ct = influence_ft * Ct_1 + influence_it * influence_gt
          ht = ot * np.tanh(Ct)
          influence_ht = influence_ot * (influence_ct/Ct) * ht

          influence_h_t_value.append(influence_ht)

          ht_1 = ht  # hidden state, previous memory state
          Ct_1 = Ct  # cell state, previous carry state

          h_t_value.append(ht)

      influence_h_t_value.append(h_t_value[-1])
      for i in range(len(influence_h_t_value)-1,0,-1):
          influence_h_t_value[i] = influence_h_t_value[i] - influence_h_t_value[i-1]

      influence_h_t_value = influence_h_t_value[1:]
      impact_columns = np.dot(influence_h_t_value,self.weights[3]) + (self.weights[4]/self.units)
      print("\n문장 : " , test_df.loc[number, '문장'])
      
      #score = loaded_model.predict(test_li[number:number+1])[0][0]
      score = self.loaded_model.predict(test_li[number:number+1])[0][0]
      test_df.loc[number,"score"] = score
      if score > standard_score:
          ment = "PASS SENTENCE"
          b_color = 'azure' #green일 경우 허용 0.5넘었을 때
          t_color = "mediumblue"
      else:
          ment = "DANGER SENTENCE"
          b_color ='mistyrose'
          t_color = "red"

      fig = plt.figure(figsize=(15,3),facecolor=b_color)
      for k in range(len(test_df.loc[number,'ko_processing'])):
          s = test_df.loc[number,'ko_processing'][k]
          k1=len(impact_columns)+k-len(test_df.loc[number,'ko_processing'])
          print(k1)
          va = round(float(impact_columns[k1]),2)
          
          if va > 0.5:
              font1 = {'family':best_font,
                  'color':  'darkblue',
                  'weight': 'normal',
                  'size': 16}
          elif va< -0.3:
              font1 = {'family':best_font,
                  'color':  'red',
                  'weight': 'normal',
                  'size': 16}

          else:
              font1 = {'family':best_font,
                  'color':  'black',
                  'weight': 'normal',
                  'size': 16}


          if k < 17:
              plt.rcParams['axes.unicode_minus'] =False
              plt.rc('font', family='NanumGothic')
              plt.text(s=s, x=k*0.7, y=0,fontdict=font1,va='center',ha='center')
              plt.text(s=va,x=k*0.7, y=-0.1,fontdict=font1,va='center',ha='center')
          elif k < 34:
              plt.rcParams['axes.unicode_minus'] =False
              plt.rc('font', family='NanumGothic')
              plt.text(s=s, x=k*0.7 - 17*0.7, y=-0.2,fontdict=font1,va='center',ha='center')
              plt.text(s=va,x=k*0.7- 17*0.7, y=-0.3,fontdict=font1,va='center',ha='center')
          else:
              plt.rcParams['axes.unicode_minus'] =False
              plt.rc('font', family='NanumGothic')
              plt.text(s=s, x=k*0.7 - 34*0.7, y=-0.4,fontdict=font1,va='center',ha='center')
              plt.text(s=va,x=k*0.7- 34*0.7, y=-0.5,fontdict=font1,va='center',ha='center')   

      plt.xlim(0,8)
      plt.ylim(-0.5,0.1)
      plt.axis('off')
      plt.title(ment, size = 20, color = t_color, pad = 15)
      plt.show()
      print("위험도 : {:.3f}".format(1-score))


a = diamond("배가 아파요.")