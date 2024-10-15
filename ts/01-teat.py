
from nixtla import NixtlaClient
import pandas as pd
from plotnine import *
from sklearn.metrics import mean_absolute_percentage_error


nk = 'nixtla-tok-9ZBl4J23XJkSBzrLQZhyI4xVP6xBLJRFuXNt5fKcwlNTvPNf6fz1h0YEi8vK3OIppQBG1GjxmFTmNw6l'

nixtla_client = NixtlaClient(api_key = nk)

#check the key
nixtla_client.validate_api_key()

# read data
df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv')
df.head()
type(df)
df.dtypes

# convert time stamp to date time
df['timestamp'] = pd.to_datetime(df.timestamp)

# plot 
(
    ggplot(df) +
    geom_line(aes('timestamp', 'value'))
)

# trn and tst
trn = df[df['timestamp'].dt.year < 1960]
tst = df[df['timestamp'].dt.year == 1960]

# Forecast
fc = nixtla_client.forecast(
    df=trn,
    h=12,
    finetune_steps=5,
    time_col='timestamp',
    target_col="value"
)

# plot forecast 
(
    ggplot(df) +
    geom_line(aes('timestamp', 'value'), color = 'darkgreen') +
    geom_line(aes('timestamp', 'TimeGPT'), color = 'red', data = fc)
    
)

# Accuracy
mape  = round(100*mean_absolute_percentage_error(tst.value, fc.TimeGPT), 2)
print ('MAPE:', mape)




# non grafica
pl = nixtla_client.plot(df, time_col='timestamp', target_col='value')
pl.show()

fc = nixtla_client.forecast(df=df, h=12, freq='MS', time_col='timestamp', target_col='value')
fc.head()

nixtla_client.plot(df, fc, time_col='timestamp', target_col='value')


