import numpy as np
import pandas as pd 
import pickle
import xgboost as xgb

from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import MinMaxScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier  

#Import data from csv
df  = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv', sep=",")

#**************** CLEAN DATA *******************
#Eliminating irrelevant data, not use
drop_cols = ['PassengerId','Cabin', 'Ticket', 'Name']
df.drop(drop_cols, axis = 1, inplace = True)

#Dropping data with fare above 300
df.drop(df[(df['Fare'] > 300)].index, inplace=True)

# Handling Missing Values in train_data

## Fill missing AGE with Median of the survided and not survided is the same
df['Age'].fillna(df['Age'].median(), inplace=True)

## Fill missing EMBARKED with Mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encoding

# Encoding the 'Sex' column
df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)

# Encoding the 'Embarked' column
df['Embarked'] = df['Embarked'].map({'S' : 0, 'C': 1, 'Q': 2})

scaler = MinMaxScaler()
train_scaler = scaler.fit(df[['Age','Fare']])
df[['Age','Fare']] = train_scaler.transform(df[['Age','Fare']])

#***************** MODEL *******************

X = df[list(df.columns[1:9])]
y = df[['Survived']]
#Split the data

#***************** MODEL GRADIENT BOOSTING - sklearn*******************


filename1 = '../models/finalized_model.sav' #use absolute path
loaded_model1 = pickle.load(open(filename1, 'rb'))


#***************** MODEL BOOSTING - XGB*******************

filename = '../models/xgb_model.sav' #use absolute path
loaded_model = pickle.load(open(filename, 'rb'))



