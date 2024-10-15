# Ref: https://github.com/mounalab/Multivariate-time-series-forecasting-keras

import pandas as pd
import numpy as np
import datetime as dt

from  plotnine import * 
from nixtla import NixtlaClient
from sklearn.metrics import mean_absolute_percentage_error
from utilsforecast.preprocessing import fill_gaps


nk = 'nixtla-tok-9ZBl4J23XJkSBzrLQZhyI4xVP6xBLJRFuXNt5fKcwlNTvPNf6fz1h0YEi8vK3OIppQBG1GjxmFTmNw6l'

nixtla_client = NixtlaClient(api_key = nk)


#####################################



file = './data/household_power_consumption.txt'

d_type = {
'date':'object',
'time':'object',
'global_active_power':'float64',
'global_reactive_power':'float64',
'voltage':'float64',
'global_intensity':'float64',
'sub_metering_1':'float64',
'sub_metering_2':'float64',
'sub_metering_3':'float64'}


df = pd.read_csv(file, sep = ';', 
                skiprows = 1, 
                names = d_type.keys(), dtype = d_type,   na_values = ['?'])

df = df.rename(columns = {'global_active_power':'value'})


df['timestamp'] =pd.to_datetime(df.date + ' ' + df.time)

df = df[[ 'timestamp',	'value']]
# round to teh hour 
df['timestamp'] = df['timestamp'].dt.round('60min')


# group by
df = df.groupby(by = ['timestamp'], as_index = False).mean()

# add id
df['id'] = 1

# reorder
df = df[[ 'id', 'timestamp',	'value']]

#########################################



# fill gaps
df = fill_gaps(df, freq='h', id_col = 'id', time_col = 'timestamp')

df['value'] = df['value'].interpolate(method='linear', limit_direction='both')



# plot 
(
    ggplot(df) +
    geom_line(aes('timestamp', 'value'))
)


# trn and tst
trn = df[df['timestamp'].dt.date <= dt.date(2010,11,20)]
tst = df[df['timestamp'].dt.date > dt.date(2010,11,20)]




# Forecast
fc = nixtla_client.forecast(
    df=trn,
    h=tst.shape[0],
    finetune_steps=15,
    finetune_loss="mae",
    time_col='timestamp',
    target_col="value"
)

# plot forecast 
(
    ggplot(df.tail(100)) +
    geom_line(aes('timestamp', 'value'), color = 'darkgreen') +
    geom_line(aes('timestamp', 'TimeGPT'), color = 'red', data = fc)
    
)

# Accuracy
mape  = round(100*mean_absolute_percentage_error(tst.value, fc.TimeGPT), 2)
print ('MAPE:', mape)


