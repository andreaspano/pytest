import pandas as pd
import numpy as np
from datetime import datetime
from plotnine import * 

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def rescale(x):
    x = np.double(x)
    m = x.min()
    M = np.max(x - m)
    y = (x-m)/M
    return y


### read data 
df = pd.read_csv( './data/passenger.csv', sep = ',', 
    dtype = {'month':'str', 'passengers' :'int64'})

# transform to data time
df['month'] = pd.to_datetime(df.month)

df['passengers'] = rescale(df.passengers)



(
    ggplot(df) +
    geom_line(aes('month', 'passengers'), color = 'blue')
)

# train and test 
# TODO: convert into function 
y = 'passengers'
n = df.shape[0]

# f = 0.8
# n_trn = np.int_(round(n*f, 0))
# n_tst =n-n_trn



h = 12
w = 13
n_trn = n-h+w
n_tst = h+w




trn = df.head(n_trn)[y].values
tst = df.tail(n_tst)[y].values

df_tst = df.tail(h)

#df_tst = df.tail(n_tst)
#df_trn = df.head(n_trn)

# h = 25
# n = df.shape[0]


# trn = df.head(n-h)[y].values
# tst = df.tail(h)[y].values

# df_tst = df.tail(h)
#df_trn = df.head(n-h)


w = 13
# X trn embedding 
x_trn = np.lib.stride_tricks.sliding_window_view(trn, w)[:-1]
# y trn
y_trn = trn[w:]
# X tst embedding 
x_tst = np.lib.stride_tricks.sliding_window_view(tst, w)[:-1]
# y tst
y_tst = tst[w:]


model = Sequential()
model.add(SimpleRNN(32, input_shape=(w,1),  activation='tanh'))
model.add(Dense(units=1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')
    
model.fit(x_trn, y_trn, epochs=10, batch_size=1, verbose=2)


#prd = model.predict(x_tst)



df_tst['prd'] = model.predict(x_tst)
 



(
    ggplot(df)+
        geom_line(aes('month', 'passengers'), color = 'green')+
        geom_line(aes('month', 'prd'), color = 'blue', data = df_tst) 

 ) 
 



#########################################
# def create_RNN(hidden_units, dense_units, input_shape, activation):
#     model = Sequential()
#     #model.add(Dense(8,  kernel_initializer='normal', activation='relu'))
#     model.add(SimpleRNN(hidden_units, input_shape=input_shape,  activation=activation[0]))
#     #model.add(LSTM(4, input_shape=(1, w)))activation[0]
#     model.add(Dense(units=dense_units, activation=activation[1]))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model
 
# model = create_RNN(hidden_units=36, dense_units=1, input_shape=(w,1), activation=['tanh', 'tanh'])
# model.fit(x_trn, y_trn, epochs=100, batch_size=1, verbose=2)

