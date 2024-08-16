import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay




#it is a magic function that renders the figure in a notebook (instead of displaying a dump of the figure object).
#%matplotlib inline

#import os
#print(os.listdir())

import warnings
warnings.filterwarnings('ignore')


col_spec = {'age': 'int64',
            'sex': 'int64',
            'pain': 'int64',
            'pressure': 'int64',
            'cholestoral': 'int64',
            'sugar': 'int64',
            'resting': 'int64',
            'heart_rate': 'int64',
            'angina': 'float64',
            'oldpeak': 'int64',
            'slope': 'int64',
            'vessels': 'int64',
            'thal': 'int64', 
            'target': 'int64'}



df = pd.read_csv("./data/heart.csv",sep = '\s', dtype = col_spec, names =  col_spec.keys())
df.target = df.target-1

   
                
df.info()
df.head()
df.describe()

#trn=df.sample(frac=0.8,random_state=200)
#trn_x = trn.drop('target', axis = 1)
#trn_y = trn['target']

#tst=df.drop(trn.index)
#tst_x = tst.drop('target', axis = 1)
#tst_y = tst['target']


from sklearn.model_selection import train_test_split
X = df.drop("target",axis=1)
y = df["target"]
trn_x,tst_x,trn_y,tst_y = train_test_split(X,y,test_size=0.20,random_state=0)


# Logistic
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(trn_x,trn_y)

prd_lr = lr.predict(tst_x)
prd_lr.shape
score_lr = round(accuracy_score(prd_lr,tst_y)*100,2)

print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")

##########################################################
# Keras snn
model = Sequential()
model.add(Dense(11,activation='relu',input_dim=13))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


model.fit(trn_x,trn_y,epochs=2000)




prd_nn = model.predict(tst_x)
prd_nn.shape


rounded = [round(x[0]) for x in prd_nn]


prd_nn = rounded
score_nn = round(accuracy_score(prd_nn,tst_y)*100,2)

print("The accuracy score achieved using Neural Network is: "+str(score_nn)+" %")

cm = confusion_matrix(prd_nn, tst_y)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels= tst_y.unique())


disp.plot()
plt.show()








