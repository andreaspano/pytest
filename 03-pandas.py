import pandas as pd
print ( pd.__version__) 



d = {
  'cars': [" BMW", "Volvo", "Ford"],
  'passings': [3, 7, 2],
  'test': [True, False, True]
}

type(d)
df = pd.DataFrame(d, index = ['x','y','z'])


print(df)

# select columns
df.cars
df['cars']
df[['cars','test']]
df[2]
    

# filter rows
# a cosa servono i due punti?
df.iloc[1]
df.iloc[1,:]
df.iloc[[0,1],:]
df.iloc[[0,1]]

df.loc['x']
df.loc[['x','y']]
df.loc[['x','y'],:]

# filter and select 
df.loc['x', 'cars']
df.loc[['x', 'z'], ['cars', 'test']]

df[df.cars == 'BMW']
df[(df.cars == 'BMW') | (df.cars == 'Volvo')]

# attributes
# no brackets
df.shape


#data types 
df.info()

# srings 
df['cars'] = df['cars'].str.strip()
print(df)

#factors
df.test = df.test.astype('category')
df.info()


df.test.cat.categories
df.test.cat.codes

#############################


d = {'name': ['a', 'b', 'c'], 
     'date': ['2014-01-12', '2014-02-15', '2025-04-12']
     }

df = pd.DataFrame(d)
df['date'] = pd.to_datetime(df.date)
df.info()

df.date.dt.date
df.date.dt.day
df.date.dt.month
df.date.dt.month_name
##########################
# melt and pivot 

import numpy as np


df = pd.DataFrame({
  'name': ['A', 'B', 'C'],
  'TA': [np.nan, 12, 77], 
  'TB': [55, 12, 77], 
})

df1 = pd.melt(df, id_vars = 'name')

df2 = pd.pivot_table(df1, index = 'name', columns = 'variable', values = 'value' )

# reset index ?
df3 = df2.reset_index()

df2.info()
df3.info()


# group by
df1.groupby('name').value.mean()

#######################

#test 

df = pd.read_csv('~/tmp/mtcars.csv')

# select 
df  = df[['mpg', 'cyl', 'disp' , 'gear' , 'carb']]


# filter 
df.loc[df.mpg > 21]
df.loc[(df.mpg > 21) & (df.disp < 100)]

# summarise

d1 = df.disp.mean()
type(d1)

#okkio che non ritorna un df
m =   df.disp.mean()


# groupby
df.groupby('carb').disp.mean().reset_index()

df.groupby('carb')['disp'].mean()

# reset_index()
df.groupby('carb').disp.mean().reset_index()


dd = df.groupby('carb', as_index=False).agg({'disp': ['mean', 'max'], 'mpg': 'mean'})

type(dd)
dd.columns
#####
import matplotlib.pyplot as plt

f,a = plt.subplots()
pl = df.mpg.plot(kind = 'hist', ax = a)
f.show()



carb_count = df.carb.value_counts()
carb_count.plot(kind = 'bar')
plt.show()


df.plot(kind = 'scatter', x = 'mpg', y = 'disp')
plt.show()



df.boxplopt(by = 'carb', column = 'mpg')
plt.show()

## Seaborn 
import seaborn as sns
import matplotlib.pyplot as plt

fig = sns.histplot(x = 'mpg', data = df, color = 'green')
plt.xlabel('Miles per gallon')
fig.set_ylabel("Counts")
plt.show()

sns.countplot(x = 'carb', data = df)
plt.show()

sns.boxplot(x = 'carb', y = 'mpg', data = df)
plt.show()


sns.regplot(x = 'mpg', y = 'disp', data = df)
sns.regplot(x = 'mpg', y = 'disp', data = df, fit_reg=False)

sns.lmplot(x = 'mpg', y = 'disp', data = df, fit_reg=False)
sns.lmplot(x = 'mpg', y = 'disp', data = df, fit_reg=False, col = 'carb')

  

g = sns.FacetGrid(data = df, col = 'gear')
g = g.map(sns.lmplot, x = 'mpg', y = 'disp')

sns.histplot(data = df, x = 'mpg', col = 'gear')

################################
import plotly.express as px
df['gear'] = df['gear'].astype('category')
fig = px.scatter(df, x = 'mpg', y = 'disp', color = 'gear')

fig = fig.update_traces(marker_size=30)
fig = fig.update_layout(scattermode="group", scattergap=0.75)
fig.show()


df.head()

import plotly as pl

# vbasic 
fig = pl.plot(df, x = 'mpg', y = 'disp', kind = 'scatter')

# aggiorna il size dei punti
fig = fig.update_traces(marker=dict(size=30))

# axix
fig = fig.update_xaxes(title = dict(text = 'Miles per Gallon', font_size=33, font_color = 'red'))

# color by group 
fig = pl.plot(df, x = 'mpg', y = 'disp', kind = 'scatter', color = 'gear')
fig = fig.update_traces(marker=dict(size=30))
fig = fig.update_xaxes(title = dict(text = 'Miles per Gallon', font_size=20))
fig = fig.update_yaxes(title = dict(text = 'Displacement', font_size=20))


# add reg line
fig = pl.plot(df, x = 'mpg', y = 'disp', kind = 'scatter', trendline = 'ols')
fig = fig.update_traces(line=dict(width=12, color = 'red'))

# facets
fig = pl.plot(df, x = 'mpg', y = 'disp', kind = 'scatter', facet_row = 'carb', facet_col_wrap= 2)

fig.show()




import plotly.express as px
df = px.data.gapminder()
fig = px.scatter(df, x='gdpPercap', y='lifeExp', color='continent', size='pop',
                facet_col='year', facet_col_wrap=4)
fig.show()




