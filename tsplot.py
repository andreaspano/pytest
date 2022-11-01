import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sktime.forecasting.arima import AutoARIMA

sizets = (8,4.5)

# import all data 
dfa = pd.read_csv('/data/hera/data_complete.csv', sep = ',',  parse_dates=['giorno_gas'])

df = dfa
df = df[df.pdr_id == 3081000818620]
df = df[['giorno_gas', 'volume_giorno']]
df = df.sort_values(by = 'giorno_gas')
df = df.set_index('giorno_gas')
# df = df.squeeze()


# plot 
font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 12}
plt.rc('font', **font)
plt.figure()
plt.plot(df.volume_giorno)
# non metto x nel grafico perche' c'e' ilindice
# plt.plot(df.giorno_gas, df.volume_giorno)
plt.xlabel("Giorno Gas")
plt.ylabel("Volume Gas")
plt.grid()
plt.show()


# split trn and tst 
str_trn = '2019-01-01'
end_trn = '2021-12-31'
str_tst = '2022-01-01'
end_tst = '2022-03-31'

pd.options.mode.chained_assignment = None

trn = df[(df.index >= str_trn) &  (df.index <= end_trn) ]
trn.index.freq = 'd' 
trn['volume_giorno'] = trn['volume_giorno'].fillna(0)
tst = df[(df.index >= str_tst) &  (df.index <= end_tst) ]

# plot 
font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 12}
plt.rc('font', **font)
plt.figure()
plt.plot(trn.volume_giorno)
plt.plot(tst.volume_giorno)
# non metto x nel grafico perche' c'e' ilindice
# plt.plot(df.giorno_gas, df.volume_giorno)
plt.xlabel("Giorno Gas")
plt.ylabel("Volume Gas")
plt.grid()
plt.show()





from sktime.forecasting.arima import AutoARIMA
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error


# fh = np.arange(len(y_test)) + 1  # forecasting horizon

forecaster = AutoARIMA(sp=1, suppress_warnings=True)
forecaster.fit(trn.volume_giorno)
prd = forecaster.predict(tst.index)

font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
plt.figure()
plt.plot(trn.volume_giorno)
plt.plot(tst.volume_giorno)
plt.plot(prd)
# non metto x nel grafico perche' c'e' ilindice
# plt.plot(df.giorno_gas, df.volume_giorno)
plt.xlabel("Giorno Gas")
plt.ylabel("Volume Gas")
plt.grid()
plt.show()


print ( 'MAPE' , mean_absolute_percentage_error(prd, tst))


