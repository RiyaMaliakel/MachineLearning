"""This module trains a model to recognize temperature patterns and
predict causes using Convolution 1D model
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.models import load_model

import pandas as pd
import numpy as np

def read_data():
    """read_data() function reads the temperature data from csv file to
    form input and output array and performs one hot encoding of target output
    """
    df_train = pd.read_csv('./dataset/train_data_one_to_one.csv')
    df_x = df_train.iloc[:, 1:101]
    x_train = np.array(df_x)
    x_train = np.expand_dims(x_train, axis=2)
    df_y_train = df_train.iloc[:, 101]
    df_test = pd.read_csv('./dataset/test_data.csv')
    df_x_test = df_test.iloc[:, 1:101]
    x_test = np.array(df_x_test)
    x_test = np.expand_dims(x_test, axis=2)
    df_y_test = df_test.iloc[:, 101]
    y_df_all = df_y_train.append(df_y_test, ignore_index=True)
    y_encode = pd.get_dummies(y_df_all)
    s_2 = y_encode.idxmax()
    print(s_2)
    print(y_encode[:10])
    print(y_encode[630:])
    y_all = np.array(y_encode)
    y_train = y_all[:580]
    y_test = y_all[580:]
    return x_train, y_train, x_test, y_test

def create_model(x_train, y_train, x_test, y_test):
    """create_model() function creates a model 1D Convolution layers"""
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(100, 1)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=16, epochs=10)
    score = model.evaluate(x_test, y_test, batch_size=1)
    print('Model saved: ', score)
    model.save('conv1d_model.h5')
    return model

#def load_saved_model():
#    """ To return the saved 1D convolution model"""
#    return load_model('conv1d_model.h5')

x_train, y_train, x_test, y_test = read_data()
model = create_model(x_train, y_train, x_test, y_test)
#model=load_saved_model()
score = model.evaluate(x_test, y_test, batch_size=1)
print(score)
print(type(x_test))
result = model.predict(x_test, batch_size=1)

print(x_test)
result1=model.predict(np.array([[29.17471568,29.76405684	,30.84482463,	30.32339391,	29.73329031,
                      30.04066685	,30.74194659,	30.25092148,	29.6481149,	29.78398576,	
                      29.44049608,	30.42494734,	29.8298252,	30.07808228	,29.77551232,
                      27.337296,	30.40738712,	29.9320071	,30.26984677,	29.44525899	,
                      30.36537878,	30.77201624	,29.83361093,	29.72099028	,28.65408974	,
                      30.13280888	,30.25031467	,29.96891504	,29.48724756	,29.61136307,	
                      29.98093748,	30.46535166,	29.85213395,	29.45589858,	30.25467641,	
                      30.55215087,	29.4992104	,29.87112633,	29.80089984	,29.20870178	,
                      29.74376399,	30.33484934,	30.8231989	,29.4300693,	29.86908799,
                      29.6191701,	29.91692151,	30.1770015	,30.64518861	,29.96955381,
                      30.7169073,	30.36864912	,30.73827504,	29.7723913	,29.68167733,
                      29.3330285,	28.69097815,	29.82367845,	30.68329403	,29.607483,
                      30.32338649,	30.51421384,	30.30619298,	30.07369326,	29.92504874,
                      30.5590828,	30.66255272,	30.26004771,	30.08813001,	29.88002286,
                      26,	25,	25	,27,	27,
                      26,	25,	25,	27,	21.64596788,
                      21.37790812,	22.71692561,	22.27252918	,22.98197372,	21.69462767, 
                      21.52714713,	21.51106944,	22.72719788,	22.71152529,	21.66094746	
                      ,21.61024023,	21.62752464,	21.97324834,	22.37603146,	21.33221417,
                      22.4078944,	22.06440349,	21.78198183,	23.01542598,23.3976021],
    [29.17471568,29.76405684	,30.84482463,	30.32339391,	29.73329031,
                      30.04066685	,30.74194659,	30.25092148,	29.6481149,	29.78398576,	
                      29.44049608,	30.42494734,	29.8298252,	30.07808228	,29.77551232,
                      27.337296,	30.40738712,	29.9320071	,30.26984677,	29.44525899	,
                      30.36537878,	30.77201624	,29.83361093,	29.72099028	,28.65408974	,
                      30.13280888	,30.25031467	,29.96891504	,29.48724756	,29.61136307,	
                      29.98093748,	30.46535166,	29.85213395,	29.45589858,	30.25467641,	
                      30.55215087,	29.4992104	,29.87112633,	29.80089984	,29.20870178	,
                      29.74376399,	30.33484934,	30.8231989	,29.4300693,	29.86908799,
                      29.6191701,	29.91692151,	30.1770015	,30.64518861	,29.96955381,
                      30.7169073,	30.36864912	,30.73827504,	29.7723913	,29.68167733,
                      29.3330285,	28.69097815,	29.82367845,	30.68329403	,29.607483,
                      30.32338649,	30.51421384,	30.30619298,	30.07369326,	29.92504874,
                      30.5590828,	30.66255272,	30.26004771,	30.08813001,	29.88002286,
                      26,	25,	25	,27,	27,
                      26,	25,	25,	27,	21.64596788,
                      21.37790812,	22.71692561,	22.27252918	,22.98197372,	21.69462767, 
                      21.52714713,	21.51106944,	22.72719788,	22.71152529,	21.66094746	
                      ,21.61024023,	21.62752464,	21.97324834,	22.37603146,	21.33221417,
                      22.4078944,	22.06440349,	21.78198183,	23.01542598,23.3976021]]))
indexes = np.argmax(result, axis=1)
print(indexes)
array = ['Anemometer errors', 'BladeAccumulatorPressureIssues', 'CoolingSystemIssues',
         'Generator speed discrepancies', 'Oil Leakage', 'Overheated oil']
res = []
for i in  indexes:
    res = np.append(res, array[i])
print(list(res))

#model.save('conv1d_save.h5')
print(result)

#import dill as pickle
#filename = 'model_v1.pk'
#with open('C:\\Users\\A661242\\flask_tutorial'+filename, 'wb') as file:
#	pickle.dump(model, file)