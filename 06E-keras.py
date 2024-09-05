# sonar data [208,61]
# binary classification 
# Cross Validation 


# imports
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.preprocessing import LabelEncoder


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


# read data into pandas 
df = pd.read_csv("./data/sonar.csv", header=None)


y = df.iloc[:, 60]
X = df.iloc[:,0:60]

# transform X into array
X = X.values

...
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)


trn_x,tst_x,trn_y,tst_y = train_test_split(X,y,test_size=0.20,random_state=0)

trn_x.shape
X.shape

# Model setup
model = Sequential()
model.add(Dense(60, input_shape=(60,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# model fit
fit = model.fit(trn_x,trn_y,epochs=120,  validation_split=0.2, shuffle=True)







# plots
plt.plot(fit.history['accuracy'])
plt.plot(fit.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# predict
prd_nn = model.predict(tst_x)
prd_nn = prd_nn.flatten()
prd = prd_nn > 0.5
prd = prd.astype(int)


def model_summary(prd, tst):
       # accuracy
       ac = round(accuracy_score(prd,tst_y)*100,2)\

       # precison
       pr = round(precision_score(prd,tst_y)*100,2)\

       # recall 
       rc = round(recall_score(prd,tst_y)*100,2)
       
       
       out = {'accuracy':ac, 'Precision':pr, 'Recall':rc}
       
       return out


model_summary(prd, tst_y)





# confusion Matrix
cm = confusion_matrix(prd, tst_y)


# Display 
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= [0,1])

disp.plot()
plt.show()




