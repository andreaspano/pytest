
from warnings import  filterwarnings
filterwarnings('ignore')

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

h = 12
w = 15
# n_trn = n-h+w
# n_tst = h+w


trn = df.head(n-h+w)[y].values
tst = df.tail(h+w)[y].values

df_tst = df.tail(h)


# X trn embedding 
x_trn = np.lib.stride_tricks.sliding_window_view(trn, w)[:-1]
# y trn
y_trn = trn[w:]
# X tst embedding 
x_tst = np.lib.stride_tricks.sliding_window_view(tst, w)[:-1]
# y tst
y_tst = tst[w:]


model = Sequential()
# simpleRNN expets a 3D tensor [batch, timesteps, feature]
model.add(SimpleRNN(32, input_shape=(w,1),  activation='tanh'))
model.add(Dense(units=1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
    
model.fit(x_trn, y_trn, epochs=10, batch_size=1, verbose=2)

df_tst['prd'] = model.predict(x_tst)
 



(
    ggplot(df)+
        geom_line(aes('month', 'passengers'), color = 'green')+
        geom_line(aes('month', 'prd'), color = 'blue', data = df_tst) 

 ) 
 


