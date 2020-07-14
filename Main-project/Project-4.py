#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


loan=pd.read_excel("Project - 4 - Train Data.xlsx")
loan=loan.drop(['Loanapp_ID','first_name','last_name','email','address','INT_ID','Prev_ID','AGT_ID'],axis=1)
loan.head()


# In[3]:


loan.info()


# In[4]:


loan.isnull().values.sum()


# In[5]:


loan['Sex'].value_counts()


# In[6]:


loan['Marital_Status'].value_counts()


# In[7]:


loan['Dependents'].value_counts()


# In[8]:


loan['Qual_var'].value_counts()


# In[9]:


loan['SE'].value_counts()


# In[10]:


loan['Prop_Area'].value_counts()


# In[11]:


loan['CPL_Status'].value_counts()


# In[12]:


loan.isnull().sum()


# In[13]:


loan.isnull().sum().sum()


# In[14]:


loan.CPL_Amount.fillna(loan['CPL_Amount'].mean(),inplace=True)
loan.CPL_Term.fillna(loan['CPL_Term'].mean(),inplace=True)
loan.Credit_His.fillna(loan['Credit_His'].mean(),inplace=True)
loan.isnull().sum()


# In[15]:


loan.info()


# In[16]:


loan['Sex']=loan.Sex.map({'M':1,'F':0})
loan['Marital_Status']=loan.Marital_Status.map({'Y':1,'N':0})
loan['Dependents']=loan.Dependents.map({0:0,1:1,2:2,'3+':3})
loan['SE']=loan.SE.map({'Y':1,'N':0})
loan['Qual_var']=loan.Qual_var.map({'Grad':1,'Non Grad':0})
loan['CPL_Status']=loan.CPL_Status.map({'Y':1,'N':0})

prop_area=pd.get_dummies(loan['Prop_Area'],prefix='Prop_Area',drop_first=True)
loan=pd.concat([loan,prop_area],axis=1)
loan=loan.drop('Prop_Area',axis=1)


# In[17]:


loan.head(20)


# In[18]:


mode=loan.mode(axis=0)
print(mode)
loan['Sex'].fillna(mode.iloc[0,0],inplace=True)
loan['Marital_Status'].fillna(mode.iloc[0,1],inplace=True)
loan['Dependents'].fillna(mode.iloc[0,2],inplace=True)
loan['SE'].fillna(mode.iloc[0,4],inplace=True)


# In[19]:


loan.isnull().sum()


# In[20]:


plt.figure(figsize=(20,10))
sns.heatmap(loan.corr(),annot=True)


# In[21]:


X_train=loan.drop('CPL_Status',axis=1)
Y_train=loan['CPL_Status']


# # Test data preparation

# In[22]:


loan1=pd.read_excel("Project - 4 - Test Data.xlsx")
loan1=loan1.drop(['Loanapp_ID','first_name','last_name','email','address','INT_ID','Prev_ID','AGT_ID'],axis=1)
loan1.head()


# In[23]:


loan1.info()


# In[24]:


loan1.isnull().values.sum()


# In[25]:


loan1['Sex'].value_counts()


# In[26]:


loan1['Marital_Status'].value_counts()


# In[27]:


loan1['Dependents'].value_counts()


# In[28]:


loan1['Qual_var'].value_counts()


# In[29]:


loan1['SE'].value_counts()


# In[30]:


loan1['Prop_Area'].value_counts()


# In[31]:


loan1.isnull().sum()


# In[32]:


loan1.isnull().sum().sum()


# In[33]:


loan1.CPL_Term.fillna(loan1['CPL_Term'].mean(),inplace=True)
loan1.Credit_His.fillna(loan1['Credit_His'].mean(),inplace=True)
loan1.isnull().sum()


# In[34]:


loan1['Sex']=loan1.Sex.map({'M':1,'F':0})
loan1['Marital_Status']=loan1.Marital_Status.map({'Y':1,'N':0})
loan1['Dependents']=loan1.Dependents.map({0:0,1:1,2:2,'3+':3})
loan1['SE']=loan1.SE.map({'Y':1,'N':0})
loan1['Qual_var']=loan1.Qual_var.map({'Grad':1,'Non Grad':0})

prop_area1=pd.get_dummies(loan1['Prop_Area'],prefix='Prop_Area',drop_first=True)
loan1=pd.concat([loan1,prop_area1],axis=1)
loan1=loan1.drop('Prop_Area',axis=1)


# In[35]:


loan1.head()


# In[36]:


mode1=loan1.mode(axis=0)
print(mode1)
loan1['Sex'].fillna(mode1.iloc[0,0],inplace=True)
loan1['Dependents'].fillna(mode1.iloc[0,2],inplace=True)
loan1['SE'].fillna(mode1.iloc[0,4],inplace=True)


# In[37]:


loan.isnull().sum().sum()


# In[38]:


X_test=loan1


# # Model Building

# # Logistic Regression

# In[39]:


from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import scale


# In[40]:


X_train=scale(X_train)
X_test=scale(X_test)


# In[41]:


LR=LogisticRegression(max_iter=200)
kf=KFold(n_splits=5,shuffle=True,random_state=10)
accuracy=cross_val_score(LR,X_train,Y_train,cv=kf,scoring='accuracy')
print('Accuracy=',accuracy)
print('Average accuracy =',accuracy.mean())


# In[42]:


Score_Table=pd.DataFrame(columns=['Model name','Accuracy(%)'])
Score_Table.loc[0]=['Logistic Reg',accuracy.mean()*100]


# # Support Vector Classifier

# In[43]:


svc=SVC(C=1)
svc_accuracy=cross_val_score(svc,X_train,Y_train,cv=kf,scoring='accuracy')
print('Accuracy=',svc_accuracy)
print('Average accuracy with C=1 is=',svc_accuracy.mean())


# In[44]:


from sklearn.model_selection import GridSearchCV
params={"C":[0.1,1,10,100,1000]}
Gridsearch=GridSearchCV(estimator=svc,param_grid=params,scoring='accuracy',cv=kf,verbose=1,return_train_score=True)


# In[45]:


Gridsearch.fit(X_train,Y_train)
results=pd.DataFrame(Gridsearch.cv_results_)
results


# In[46]:


plt.figure(figsize=(6,6))
plt.plot(results['param_C'],results['mean_test_score'])
plt.plot(results['param_C'],results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('score')
plt.legend(['test score','train score'], loc='upper right')
plt.xscale('log')
plt.show()


# In[47]:


params={"C":[0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,2]}
Gridsearch=GridSearchCV(estimator=svc,param_grid=params,scoring='accuracy',cv=kf,verbose=1,return_train_score=True)
Gridsearch.fit(X_train,Y_train)
results=pd.DataFrame(Gridsearch.cv_results_)
results


# In[48]:


plt.figure(figsize=(6,6))
plt.plot(results['param_C'],results['mean_test_score'])
plt.plot(results['param_C'],results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('score')
plt.legend(['test score','train score'], loc='upper right')
plt.show()


# In[49]:


results[results.mean_test_score==results.mean_test_score.max()]


# In[50]:


print('best score for SVC=',Gridsearch.best_score_)
print('best parameter for SVC=',Gridsearch.best_params_)


# In[51]:


best_c=Gridsearch.best_params_
model=SVC(best_c['C'])
svc_accuracy1=cross_val_score(model,X_train,Y_train,cv=kf,scoring='accuracy')
print('Accuracy for SVC=',svc_accuracy1)
print('Mean Accuracy for SVC=',svc_accuracy1.mean())


# In[52]:


Score_Table.loc[1]=['SVC for C=0.5',svc_accuracy1.mean()*100]


# # K-Nearest Neighbor

# In[53]:


from sklearn import neighbors
N={"n_neighbors":[1,2,5,10,20,30,40,50,60]}
clf=neighbors.KNeighborsClassifier()
Gridsearch=GridSearchCV(estimator=clf,param_grid=N,scoring='accuracy',cv=kf,verbose=1,return_train_score=True)


# In[54]:


Gridsearch.fit(X_train,Y_train)
clf_results=pd.DataFrame(Gridsearch.cv_results_)
clf_results


# In[55]:


best_score=Gridsearch.best_score_
best_n=Gridsearch.best_params_
print('best test score=',best_score)
print('best number of neighbours=',best_n)


# In[56]:


N={"n_neighbors":[14,15,16,17,18,19,20,21,22,23,24,25]}
clf=neighbors.KNeighborsClassifier()
Gridsearch=GridSearchCV(estimator=clf,param_grid=N,scoring='accuracy',cv=kf,verbose=1,return_train_score=True)
Gridsearch.fit(X_train,Y_train)
clf_results=pd.DataFrame(Gridsearch.cv_results_)

best_score=Gridsearch.best_score_
best_n=Gridsearch.best_params_


# In[57]:


print('best accuracy score=',best_score)
print('best number of neighbours=',best_n)


# In[58]:


Score_Table.loc[2]=['KNN for neighbors =22',best_score*100]


# # Decision Tree

# In[59]:


loan=pd.read_excel("Project - 4 - Train Data.xlsx")
loan=loan.drop(['Loanapp_ID','first_name','last_name','email','address','INT_ID','Prev_ID','AGT_ID'],axis=1)
loan.head()


# In[ ]:





# In[60]:


loan.CPL_Amount.fillna(loan['CPL_Amount'].mean(),inplace=True)
loan.CPL_Term.fillna(loan['CPL_Term'].mean(),inplace=True)
loan.Credit_His.fillna(loan['Credit_His'].mean(),inplace=True)
loan.isnull().sum()


# In[61]:


loan['Sex']=loan.Sex.map({'M':1,'F':0})
loan['Marital_Status']=loan.Marital_Status.map({'Y':1,'N':0})
loan['Dependents']=loan.Dependents.map({0:0,1:1,2:2,'3+':3})
loan['SE']=loan.SE.map({'Y':1,'N':0})
loan['Qual_var']=loan.Qual_var.map({'Grad':1,'Non Grad':0})
loan['Prop_Area']=loan.Prop_Area.map({'Urban':1,'Semi U':2,'Rural':3})
loan['CPL_Status']=loan.CPL_Status.map({'Y':1,'N':0})


# In[62]:


mode=loan.mode(axis=0)
print(mode)
loan['Sex'].fillna(mode.iloc[0,0],inplace=True)
loan['Marital_Status'].fillna(mode.iloc[0,1],inplace=True)
loan['Dependents'].fillna(mode.iloc[0,2],inplace=True)
loan['SE'].fillna(mode.iloc[0,4],inplace=True)


# In[63]:


loan['Sex']=loan.Sex.map({1:'M',0:'F'})
loan['Marital_Status']=loan.Marital_Status.map({1:'Y',0:'N'})
loan['Dependents']=loan.Dependents.map({0:'0',1:'1',2:'2',3:'3+'})
loan['SE']=loan.SE.map({1:'Y',0:'N'})
loan['Qual_var']=loan.Qual_var.map({1:'Grad',0:'Non Grad'})
loan['Prop_Area']=loan.Prop_Area.map({1:'Urban',2:'Semi U',3:'Rural'})
loan['CPL_Status']=loan.CPL_Status.map({1:'Y',0:'N'})


# In[64]:


from sklearn import preprocessing
df_categorical = loan.select_dtypes(include=['object'])
le = preprocessing.LabelEncoder()
df_categorical = df_categorical.apply(le.fit_transform)
df_categorical.head()


# In[65]:


loan = loan.drop(df_categorical.columns, axis=1)
loan = pd.concat([loan, df_categorical], axis=1)
loan.head()


# Model building for DT

# In[66]:


X_train=loan.drop('CPL_Status',axis=1)
Y_train=loan['CPL_Status']


# In[67]:


from sklearn.tree import DecisionTreeClassifier
parameters = {'max_depth': range(1, 10)}

DT = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)

DT_clf = GridSearchCV(DT, parameters, 
                    cv=kf, 
                   scoring="accuracy")
DT_clf.fit(X_train, Y_train)


# In[68]:


results=pd.DataFrame(DT_clf.cv_results_)
results.head()


# In[69]:


print('best maximum depth=',DT_clf.best_params_)
print('best accuracy=',DT_clf.best_score_)


# In[70]:


parameters = {'min_samples_leaf': range(1, 100, 2)}

DT = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)

DT_clf1 = GridSearchCV(DT, parameters, 
                    cv=kf, 
                   scoring="accuracy")
DT_clf1.fit(X_train, Y_train)
print('best min_samples_leaf=',DT_clf1.best_params_)
print('best accuracy=',DT_clf1.best_score_)


# In[71]:


parameters = {'min_samples_split': range(2, 100, 2)}
DT = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)

DT_clf2 = GridSearchCV(DT, parameters, 
                    cv=kf, 
                   scoring="accuracy")
DT_clf2.fit(X_train, Y_train)
print('best min_samples_split=',DT_clf2.best_params_)
print('best accuracy=',DT_clf2.best_score_)


# In[72]:


param = {
    'max_depth': range(1, 10),
    'min_samples_leaf': range(1, 100, 10),
    'min_samples_split': range(2, 102, 10),
    'criterion': ["entropy", "gini"]
}

DT = DecisionTreeClassifier()

DT_clf3 = GridSearchCV(estimator=DT, param_grid=param, 
                    cv=kf, 
                   scoring="accuracy",verbose = 1)
DT_clf3.fit(X_train, Y_train)


# In[73]:


print('best parameters=',DT_clf3.best_params_)
print('best accuracy=',DT_clf3.best_score_)


# In[74]:


Score_Table.loc[3]=['DT (max depth=1, min_samples_leaf=1, min_samples_split=2)',(DT_clf3.best_score_)*100]


# In[76]:


Score_Table


# # Stacking 

# In[77]:


from sklearn.ensemble import StackingClassifier


# In[89]:


from sklearn.model_selection import train_test_split


# In[86]:


level0 = list()
level0.append(('LR', LogisticRegression(max_iter=500)))
level0.append(('clf', neighbors.KNeighborsClassifier(n_neighbors=22)))
level0.append(('DT', DecisionTreeClassifier(criterion='entropy', max_depth= 1, min_samples_leaf= 1, min_samples_split= 2))) 
level0.append(('svc', SVC(C=0.5)))
level1 = LogisticRegression(max_iter=500)
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)


# In[90]:


x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,train_size=0.9,random_state=100)


# In[91]:


model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[92]:


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)


# In[ ]:




