import numpy as np
from keras.models import Sequential
from keras.layers import Dense


x_train_data = np.random.random((1000, 10))
y_train_data = np.random.randint(2, size=(1000, 1))
type(y_train_data)

# Building the model
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=10))
model.add(Dense(1, activation='sigmoid'))


# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train_data, y_train_data, epochs=20, batch_size=10)

# Generate some dummy test data
x_test_data = np.random.random((100, 10))
y_test_data = np.random.randint(2, size=(100, 1))


# Evaluating the model on the test data
loss, accuracy = model.evaluate(x_test_data, y_test_data)
print('Test model loss:', loss)
print('Test model accuracy:', accuracy)



## -----------------------## 
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
from keras.models import Sequential
from keras.layers import Dense



# training 
df = pd.read_csv('~/tmp/train.csv')
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
y_trn = np.array([int(i) for i in y_trn]) 

x_tst = tst[['Client_Income','Car_Owned','Bike_Owned','Active_Loan','House_Own','Child_Count','Credit_Amount']]
x_tst = x_tst.to_numpy()

y_tst = tst['Default']
y_tst = y_tst.to_numpy()
y_tst = np.array([int(i) for i in y_tst])
 


model = Sequential()
model.add(Dense(5, activation='relu', input_dim=7))
model.add(Dense(1, activation='sigmoid'))


# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_trn, y_trn, epochs=20, batch_size=10)

y_prd = model.predict(x_tst)
y_prd = np.argmax (y_prd, axis = 1)
y_prd = np.array([int(i) for i in y_prd])



loss, accuracy = model.evaluate(x_tst, y_tst)
print('Test model loss:', loss)
print('Test model accuracy:', accuracy)

pd.crosstab(y_tst, 'n')
pd.crosstab(y_prd, 'n')




confusion_matrix(y_tst, y_prd , normalize='pred')

