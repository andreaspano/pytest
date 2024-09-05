# booking data
# binary classification 
# train & test split 


# imports
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from scikeras.wrappers import KerasClassifier
#from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import StratifiedKFold
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import CategoryEncoding







d_type = {
    'booking_id': 'object', 
    'number_of_adults': 'float64', 
    'number_of_children':'float64',
    'number_of_weekend_nights':'float64', 
    'number_of_week_nights':'float64',
    'type_of_meal':'object',
    'car_parking_space':'float64', 
    'room_type':'object', 
    'lead_time':'float64',
    'market_segment_type':'object',
    'repeated':'float64', 
    'p_c':'float64',
    'p_not_c':'float64', 
    'average_price':'float64',
    'special_requests':'float64',
    'date_of_reservation':'object',
    'booking_status':'object'}



df0 = pd.read_csv('./data/booking.csv', sep = ',', dtype = d_type, names =  d_type.keys(), skiprows=1)




#'type_of_meal',
    #'car_parking_space',  
    #'p_c', 
    # 'p_not_c', 
    #'special_requests',
    #'date_of_reservation', 
    #'booking_id',
    

df = df0[['number_of_adults', 
    'number_of_children',
    'number_of_weekend_nights', 
    'number_of_week_nights', 
    'repeated', 
    'average_price', 
    'lead_time',
    'room_type',
    'market_segment_type',
    'special_requests',
    'booking_status',
    'date_of_reservation']]



#df0.date_of_reservation

   

    

# response
response = 'booking_status'
y = df.loc[:, response]
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)


# covert categ to dummies
X = df.drop(response,  axis='columns')

str_ftr = ['number_of_adults', 'number_of_children', 'special_requests']
dt_ftr = 'date_of_reservation'

X[str_ftr] = X[str_ftr].astype(str)


# date
X[dt_ftr] = pd.to_datetime(X[dt_ftr], format='%m/%d/%Y', errors='coerce' )
X['month'] = pd.DatetimeIndex(X[dt_ftr]).month
X['month'] = X['month'].astype(str)
X = df.drop(dt_ftr,  axis='columns')






X = pd.get_dummies(X, dtype= int)
X = X.values

#split
trn_x,tst_x,trn_y,tst_y = train_test_split(X,y,test_size=0.20,random_state=0)


# Model setup
#trn_x.shape[1]
model = Sequential()
model.add(Dense(6, input_dim=trn_x.shape[1] , activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model fit
fit = model.fit(trn_x,trn_y,epochs=20,  validation_split=0.2, shuffle=True)



import matplotlib.pyplot as plt
import seaborn as sns

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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score




def model_summary(prd, tst):
       # accuracy
       ac = round(accuracy_score(prd,tst_y)*100,2)\

       # precison
       pr = round(precision_score(prd,tst_y)*100,2)\

       # recall 
       rc = round(recall_score(prd,tst_y)*100,2)
       
       
       out = {'accuracy':ac, 'Precision':pr, 'Recall':rc}
       
       return out


