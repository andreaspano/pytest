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


df = df0[['number_of_adults', 
    'number_of_children',
    'number_of_weekend_nights', 
    'number_of_week_nights', 
    'repeated', 
    'average_price', 
    'booking_status']]




    #'booking_id',
    #'type_of_meal',
    #'car_parking_space', 
    # 'room_type', 
    # 'lead_time', 
    # 'market_segment_type',
    #'p_c', 
    # 'p_not_c', 
    #'special_requests',
    #'date_of_reservation', 


# response
response = 'booking_status'
y = df.loc[:, response]
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)


# X
X = df.drop(response,  axis='columns')
X = X.values

#split
trn_x,tst_x,trn_y,tst_y = train_test_split(X,y,test_size=0.20,random_state=0)


# Model setup
model = Sequential()
model.add(Dense(16, input_shape=(6,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model fit
fit = model.fit(trn_x,trn_y,epochs=100,  validation_split=0.2, shuffle=True)


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




