
## -----------------------## 
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


# training 
df = pd.read_csv('./data/train.csv')
df = df[['Client_Income','Car_Owned','Bike_Owned','Active_Loan','House_Own','Child_Count','Credit_Amount', 'Default']]
df.Client_Income = pd.to_numeric(df.Client_Income, errors = 'coerce')
df.Credit_Amount = pd.to_numeric(df.Credit_Amount, errors = 'coerce')
#len(trn) 121856

df = df.dropna()


trn=df.sample(frac=0.8,random_state=200)
tst=df.drop(trn.index)



x_trn = trn[['Client_Income','Car_Owned','Bike_Owned','Active_Loan','House_Own','Child_Count','Credit_Amount']]
x_trn = x_trn.to_numpy()

y_trn = trn['Default']
y_trn = y_trn.to_numpy()
#y_trn = np.array([int(i) for i in y_trn]) 

#y_trn = to_categorical(y_trn)
#y_test_binary = keras.utils.to_categorical(y_test, num_classes)


x_tst = tst[['Client_Income','Car_Owned','Bike_Owned','Active_Loan','House_Own','Child_Count','Credit_Amount']]
x_tst = x_tst.to_numpy()

y_tst = tst['Default']
y_tst = y_tst.to_numpy()
y_tst = np.array([int(i) for i in y_tst])
#y_tst = to_categorical(y_tst)


 


model = Sequential()
model.add(Dense(5, activation='relu', input_dim=7))
#model.add(Dense(2, ctivation='softmax'))
#model.add(Dense(2, activation='sigmoid'))

#model = Sequential()
#model.add(Dense(500, activation='relu', input_dim=8))
#model.add(Dense(100, activation='relu'))
#model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='softmax'))




# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_trn, y_trn, epochs=20, batch_size=10)

y_prd = model.predict(x_tst)

#y_prd = y_prd > 0.08
y_prd = np.array([int(i) for i in y_prd])



#y_prd = np.argmax (y_prd, axis = 1)
#y_prd = np.array([int(i) for i in y_prd])



loss, accuracy = model.evaluate(x_tst, y_tst)
print('Test model loss:', loss)
print('Test model accuracy:', accuracy)

pd.crosstab(y_tst, 'n')
pd.crosstab(y_prd, 'n')




confusion_matrix(y_tst, y_prd , normalize='pred')

