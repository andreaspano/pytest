import pandas as pd 
import numpy as np
from plotnine import *


# read data
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'
df = pd.read_csv(url, usecols=[1], engine='python')

(
    ggplot(df) +
    geom_line(aes(y = 'Sunsposts'))
)
