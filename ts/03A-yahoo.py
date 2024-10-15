import pandas as pd
import numpy as np
from datetime import datetime
from plotnine import * 

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.optimizers import RMSprop
from keras.optimizers import Adam

from sklearn.metrics import mean_absolute_percentage_error

import matplotlib.pyplot as plt


def rescale(x):
    x = np.double(x)
    m = x.min()
    M = np.max(x - m)
    y = (x-m)/M
    return y



y = 'Close'
d = 'Date'
h = 40
w = 13
e = 100
b = 8
### read data 
df = pd.read_csv( './data/yahoo.csv', sep = ',')
#dtype = {'month':'str', 'passengers' :'int64'})

# transform to data time
df[d] = pd.to_datetime(df[d])

# sort by date
df = df.sort_values(by=[d])


df[y] = rescale(df[y])



(
    ggplot(df) +
    geom_line(aes(d, y), color = 'blue')
)

# train and test 
# TODO: convert into function 
n = df.shape[0]

# n_trn = n-h+w
# n_tst = h+w


trn = df.head(n-h+w)[y].values
tst = df.tail(h+w)[y].values

df_tst = df.tail(h)
df_tst = df_tst[[d, y]]

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
model.add(SimpleRNN(16, input_shape=(w,1),  activation='tanh'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='tanh'))
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate = 0.001))
model.fit(x_trn, y_trn, epochs=e, batch_size=b, verbose=2)

# num_units=128
# embedding=4
# num_dense=32
# lr=0.001
# model = Sequential()
# model.add(SimpleRNN(units=num_units, input_shape=(w, 1), activation="relu"))
# model.add(Dense(num_dense, activation="relu"))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer=RMSprop(learning_rate=lr),metrics=['mse'])
# fit = model.fit(x_trn, y_trn, epochs=e, batch_size=1, verbose=2, validation_split=0.2, shuffle=True)
    
# plot_model(fit)
# plt.show()

df_tst['prd'] = model.predict(x_tst)
 
mape = np.round(100*mean_absolute_percentage_error(df_tst[y], df_tst['prd']), 2)



(
    ggplot(df.tail(75))+
        geom_line(aes(d, y), color = 'green')+
        geom_line(aes(d, 'prd'), color = 'blue', data = df_tst) 

 ) 
 

print('MAPE:', mape) 

# (
#     ggplot(df_tst)+
#         geom_point(aes(y, 'prd'), color = 'green') +
#         geom_abline(intercept = 0 , slope = 1)
# ) 
 
# loss_df = pd.DataFrame({'epoc' : np.arange(e),'loss':fit.history['loss'], 'val_loss': fit.history['val_loss']})

# (
#     ggplot(loss_df) +
#     geom_line(aes('epoc', 'loss'), color = 'green')+
#     geom_line(aes('epoc', 'val_loss'), color = 'red')
# )