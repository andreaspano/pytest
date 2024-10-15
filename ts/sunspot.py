import pandas as pd 
import numpy as np
from plotnine import *

names = ['id', 'year', 'month', 'day', 'date', 'dec_year', 'value', 'sn_error', 'obs_num' ]


df = pd.read_csv('./data/sunspot.csv', sep = ',',  na_values = [-1], names = names, skiprows = 1)


df = df[['date', 'year', 'value']]

(
    ggplot(df) +
    geom_line(aes('date', 'value'))
)


start = df.index[np.isnan(df.value)].max()+1
df = df[start:]

trn = df[df.year < 2019]
tst = df[df.year == 2019]

# Windiows size
w = 10


# X trn embedding 
x_trn = np.lib.stride_tricks.sliding_window_view(trn.value.tolist(), w)[:-1]
# y trn
y_trn = x_trn[w:]
# X tst embedding 
x_tst = np.lib.stride_tricks.sliding_window_view(tst.value, w)[:-1]
# y tst
y_tst = x_tst[w:]

x_trn.shape

dataset = pd.DataFrame({'x': np.arange(1, 10)})

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


 create_dataset(dataset, 1)   