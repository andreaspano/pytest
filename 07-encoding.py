import pandas as pd

# Initialize data to Dicts of series.
d = {'one': pd.Series([10, 20, 30, 40],
                      index=['a', 'b', 'c', 'd']),
     'two': pd.Series(['a', 'b', 'a', 'b'],

                      index=['a', 'b', 'c', 'd'])}


d = {'one': pd.Series([10, 20, 30, 40]), 
    'two': pd.Series(['a', 'b', 'a', 'b']),
    'three': pd.Series(['a', 'a', 'b', 'b'])}



# creates Dataframe.
df = pd.DataFrame(d)

# print the data.
print(df)

categories = ['two', 'three']


pd.get_dummies(df)

df[categories] = df[categories].astype('category')

for j , i in categories:
    df[j] = df[j].cat.set_categories(i)
    
    
    