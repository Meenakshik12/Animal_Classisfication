#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Import necessary libraries

import pandas as pd


# In[3]:


A= pd.read_csv("/Users/ashish/OneDrive/Desktop/DS/animal information.csv")


# In[4]:


A.head()


# In[6]:


A.tail(10)


# In[7]:


A.info()


# In[8]:


A.describe().T


# In[10]:


A.shape


# In[11]:


# Find the null Values

A.isna().sum()


# In[12]:


A['class_type'].value_counts()


# In[15]:


# pandas data visualization library

import matplotlib.pyplot as plt
import seaborn as sb


# In[16]:


sb.countplot(A['class_type'])


# In[17]:


for i in A.columns[:-1]:
    plt.figure(figsize=(12,6))
    plt.title("ATTRIBUTES '%s'"%i)
    sb.countplot(A[i],hue=A['class_type'])


# In[18]:


# To find correlation between deependent and independent variables

fig=plt.figure(figsize=(10,10))
sb.heatmap(A.corr(),annot=True, cmap="YlGnBu")


# In[19]:


# Feature selection

A = A.drop(['animal_name'], axis='columns')
A.head()


# In[20]:


X = A.drop(['class_type'], axis='columns')
X.head()


# In[21]:


y = A['class_type']
y.head()


# In[32]:


# Divid the data into training and testing data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# # Model 1 GaussianNB

# In[35]:


from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X_train, y_train)
pred= model.predict(X_test)


# In[36]:


from sklearn.metrics import confusion_matrix,accuracy_score
print("ACCURACY = %.2f"%accuracy_score(y_test,pred))
print("CF = ",confusion_matrix(y_test,pred))


# In[38]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X = X.apply(le.fit_transform)


# # Model 2 DecisionTreeClassifier

# In[60]:


#Unpruned

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=22,criterion="entropy")
model = dtc.fit(xtrain,ytrain)
pred = model.predict(xtest)
from sklearn.metrics import accuracy_score,confusion_matrix
print("ACCURACY = %.2f"%accuracy_score(ytest,pred))
print("CF = ",confusion_matrix(ytest,pred))


# In[61]:


#Postpruned


from sklearn.tree import DecisionTreeClassifier
dtr = DecisionTreeClassifier(random_state=27)

from sklearn.model_selection import GridSearchCV
tp = {"max_depth":range(2,18,1)}

cv = GridSearchCV(dtr,tp,cv=4)
cvmodel = cv.fit(xtrain,ytrain)
cvmodel.best_params_


# In[65]:


from sklearn.tree import DecisionTreeClassifier
dtr = DecisionTreeClassifier(random_state=27,max_depth=6)
model = dtr.fit(xtrain,ytrain)
pred = model.predict(xtest)
from sklearn.metrics import accuracy_score,confusion_matrix
print("ACCURACY = %.2f"%accuracy_score(ytest,pred))
print("CF = ",confusion_matrix(ytest,pred))


# In[63]:


#Prepruned
from sklearn.tree import DecisionTreeClassifier
dtr = DecisionTreeClassifier(random_state=27)

from sklearn.model_selection import GridSearchCV
tp = {"min_samples_split":range(2,18,1)}

cv = GridSearchCV(dtr,tp,cv=4)
cvmodel = cv.fit(xtrain,ytrain)
cvmodel.best_params_


# In[64]:


from sklearn.tree import DecisionTreeClassifier
dtr = DecisionTreeClassifier(random_state=27,max_depth=3)
model = dtr.fit(xtrain,ytrain)
pred = model.predict(xtest)
from sklearn.metrics import accuracy_score,confusion_matrix
print("ACCURACY = %.2f"%accuracy_score(ytest,pred))
print("CF = ",confusion_matrix(ytest,pred))


# # Model 3 RandomForestClassifier

# In[56]:


from sklearn.ensemble import RandomForestClassifier
rfr = RandomForestClassifier(random_state=20)
model = rfr.fit(xtrain,ytrain)
pred = model.predict(xtest)
from sklearn.metrics import accuracy_score,confusion_matrix
print("ACCURACY = %.2f"%accuracy_score(ytest,pred))
print("CF = ",confusion_matrix(ytest,pred))


# In[ ]:




