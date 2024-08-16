# sonar data [208,61]
# binary classification 
# train & test split 


# imports
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
...

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


# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(60, input_shape=(60,), activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model



# evaluate model with standardized dataset
estimator = KerasClassifier(model=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



trn_x,tst_x,trn_y,tst_y = train_test_split(X,y,test_size=0.20,random_state=0)

trn_x.shape
X.shape

# Model setup
model = Sequential()
model.add(Dense(60, input_shape=(60,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# model fit
model.fit(trn_x,trn_y,epochs=500)

# predict
prd_nn = model.predict(tst_x)
prd_nn = prd_nn.flatten()
prd = prd_nn > 0.5
prd = prd.astype(int)


# accuracy
ac = round(accuracy_score(prd,tst_y)*100,2)\

# precison
pr = round(precision_score(prd,tst_y)*100,2)\

# recall 
rc = round(recall_score(prd,tst_y)*100,2)\
    
    

print ('Accuracy is:' , ac, '\n', 
       'Precision is:' , pr, '\n', 
       'Recall is:' , rc)





# confusion Matrix
cm = confusion_matrix(prd, tst_y)


# Display 
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= [0,1])

disp.plot()
plt.show()




