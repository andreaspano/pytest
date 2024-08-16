import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('./data/train.csv')

df.info()


d_type = {
        'ID' : 'int64'  ,
        'Client_Income' : 'object', 
        'Car_Owned' : 'Int64',
        'Bike_Owned' : 'Int64',
        'Active_Loan' : 'Int64',
        'House_Own' : 'Int64',
        'Child_Count' : 'Int64',
        'Credit_Amount' : 'object', 
        'Loan_Annuity' : 'object' ,
        'Accompany_Client' : 'object', 
        'Client_Income_Type' : 'object', 
        'Client_Education' : 'object' ,
        'Client_Marital_Status' : 'object', 
        'Client_Gender' : 'object' ,
        'Loan_Contract_Type' : 'object', 
        'Client_Housing_Type' : 'object' ,
        'Population_Region_Relative' : 'object', 
        'Age_Days' : 'object' ,
        'Employed_Days' : 'object', 
        'Registration_Days' : 'object', 
        'ID_Days' : 'object' ,
        'Own_House_Age' : 'float64',
        'Mobile_Tag' : 'int64'  ,
        'Homephone_Tag' : 'int64',  
        'Workphone_Working' : 'int64',  
        'Client_Occupation' : 'object', 
        'Client_Family_Members' : 'float64',
        'Cleint_City_Rating' : 'float64',
        'Application_Process_Day' : 'float64',
        'Application_Process_Hour' : 'float64',
        'Client_Permanent_Match_Tag' : 'object', 
        'Client_Contact_Work_Tag' : 'object' ,
        'Type_Organization' : 'object' ,
        'Score_Source_1' : 'float64',
        'Score_Source_2' : 'float64',
        'Score_Source_3' : 'object' ,
        'Social_Circle_Default' : 'float64',
        'Phone_Change' : 'float64',
        'Credit_Bureau' : 'float64',
        'Default' : 'int64'  
        }


df = pd.read_csv("./data/train.csv",sep = ',', dtype = d_type, names =  d_type.keys(), skiprows=1)

df.Client_Income = pd.to_numeric(df.Client_Income, errors = 'coerce')
df.Credit_Amount = pd.to_numeric(df.Credit_Amount, errors = 'coerce')


df = df[['Client_Income','Car_Owned','Bike_Owned','Active_Loan','House_Own',
         'Child_Count','Credit_Amount', 'Client_Marital_Status','Client_Gender', 'Client_Education',
         'Client_Housing_Type','Loan_Contract_Type',
         'Default']]




         
df.info()


df = pd.get_dummies(df)

response = 'Default'

X = df.drop(response,axis=1)
y = df[response]
trn_x,tst_x,trn_y,tst_y = train_test_split(X,y,test_size=0.20,random_state=0)


##########################################################
# Keras snn
model = Sequential()
model.add(Dense(64, activation='relu',input_dim=len(df.columns)-1))
model.add(Dense(1,  activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

len(df)


model.fit(trn_x,trn_y,epochs=1000)




prd_nn = model.predict(tst_x)

type(prd_nn)



prd_nn.shape


min(prd_nn)
max(prd_nn)
np.mean(prd_nn)




rounded = [round(x[0]) for x in prd_nn]


prd_nn = rounded
score_nn = round(accuracy_score(prd_nn,tst_y)*100,2)

print("The accuracy score achieved using Neural Network is: "+str(score_nn)+" %")

cm = confusion_matrix(prd_nn, tst_y)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels= tst_y.unique())


disp.plot()
plt.show()

