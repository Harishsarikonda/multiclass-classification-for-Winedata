
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
init_data = pd.read_csv("Knight ML Assignment/Data/train.csv")
#print("Length of dataframe before duplicates are removed:", len(init_data))
parsed_data = init_data[init_data.duplicated('designation', keep=False)]
#print("Length of dataframe after duplicates are removed:", len(parsed_data))

parsed_data.dropna(how='any',inplace=True)
#print("Length of dataframe after NaNs are removed:", len(parsed_data))

#parsed_data.info()

init_data2 = pd.read_csv("Knight ML Assignment/Data/test.csv")
#print("Length of dataframe before duplicates are removed:", len(init_data2))
parsed_data2 = init_data2[init_data.duplicated('designation', keep=False)]
#print("Length of dataframe after duplicates are removed:", len(parsed_data2))

parsed_data2.dropna(how='any',inplace=True)
#print("Length of dataframe after NaNs are removed:", len(parsed_data2))

#parsed_data2.info()
l=[]
for i in parsed_data['variety']:
	if i not in l:
		l.append(i)
#print(l)


# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import math
# loading the iris dataset 
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
label=le.fit_transform(parsed_data['variety'])
review=le.fit_transform(parsed_data['review_description'])
designation=le.fit_transform(parsed_data['designation'].astype(str))
country=le.fit_transform(parsed_data['country'].astype(str))
province=le.fit_transform(parsed_data['province'].astype(str))
review_title=le.fit_transform(parsed_data['review_title'])
region_1=le.fit_transform(parsed_data['region_1'].astype(str))
region_2=le.fit_transform(parsed_data['region_2'].astype(str))
user_name=le.fit_transform(parsed_data['user_name'].astype(str))
winery=le.fit_transform(parsed_data['winery'].astype(str))
m=[]
for i in label:
	if i not in m:
		m.append(i)
#print(m)

d={}
for i in range(len(m)):
	d[m[i]]=l[i]
#print(d)

review2=le.fit_transform(parsed_data2['review_description'])
designation2=le.fit_transform(parsed_data2['designation'].astype(str))
country2=le.fit_transform(parsed_data2['country'].astype(str))
province2=le.fit_transform(parsed_data2['province'].astype(str))
review_title2=le.fit_transform(parsed_data2['review_title'])
region_12=le.fit_transform(parsed_data2['region_1'].astype(str))
region_22=le.fit_transform(parsed_data2['region_2'].astype(str))
user_name2=le.fit_transform(parsed_data2['user_name'].astype(str))
winery2=le.fit_transform(parsed_data2['winery'].astype(str))


features=list(zip(designation,province,country,review_title,region_1,region_2,parsed_data['price']))
features2=list(zip(designation2,province2,country2,review_title2,region_12,region_22,parsed_data2['price']))
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2)
from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
#knn= KNeighborsClassifier(n_neighbors=int(math.sqrt(82657)))
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100)


# Train the model using the training sets
rfc.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = rfc.predict(X_test)
#print(le.inverse_transform(y_pred))
y_pred2= rfc.predict(features2)
#print(y_pred2)
y_pred2_decoded=[]
for i in y_pred2:
	y_pred2_decoded.append(d[i])
#print(y_pred2_decoded)
#Predict Output
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


import csv

with open('Knight ML Assignment/Data/final.csv', 'a',encoding="utf-8") as file:
       parsed_data2.to_csv(file, index=False)

parsed_data2['variety']=y_pred2_decoded
parsed_data2.to_csv('Knight ML Assignment/Data/final.csv', index=False)



