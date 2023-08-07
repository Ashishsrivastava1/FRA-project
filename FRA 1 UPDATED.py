#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color= sns.color_palette()
import sklearn.metrics as metrics

import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv('C:/Users/ASHISH SRIVASTAVA/Downloads/Company(FRA).csv')


# In[3]:


df.head()


# In[4]:


df.columns= df.columns.str.replace(' ','_').str.replace('(','').str.replace(')','').str.replace('%','perc').str.replace('/','_to_')


# In[5]:


df.head()


# In[6]:


print('The number of rows (observations) is',df.shape[0],'\n''The number of columns (variables) is',df.shape[1])


# In[7]:


# Check for duplicate data
dups = df.duplicated()
print('Number of duplicate rows = %d' % (dups.sum()))
df[dups]


# In[8]:


df.info()


# In[9]:


df.duplicated().sum()


# In[10]:


df.boxplot(figsize=(16,8))
plt.xticks(rotation=90)
plt.show()


# In[11]:


df.describe()


# In[12]:


pd.options.display.float_format='{:.2f}'.format
df.describe()


# In[13]:


df.isnull().sum()


# In[14]:


df['default']= np.where((df['Networth_Next_Year']>0),0,1)


# In[15]:


df[['default','Networth_Next_Year']].head(10)


# In[16]:


df['default'].value_counts()


# In[17]:


df['default'].value_counts(normalize=True)


# In[18]:


df.isnull().sum()


# In[19]:


df.size


# In[20]:


df.isnull().sum().sum()


# In[21]:


df_X = df.drop('default', axis = 1)
df_Y = df['default']


# In[22]:


df_X


# In[23]:


Q1=df_X.quantile(0.25)
Q3=df_X.quantile(0.75)
IQR=Q3-Q1
UL=Q3+1.5*IQR
LL=Q1-1.5*IQR


# In[24]:


def mod_outlier (df):
    df= df._get_numeric_data()
    q1= df.quantile(0.25)
    q3= df.quantile(0.75)
    iqr= q3- q1
    lower_bound= q1-(1.5* iqr)
    upper_bound= q3+(1.5* iqr)
    for col in df.columns:
        for i in range(0,len(df[col])):
            if df[col][i] < lower_bound[col]:
                df[col][i] = lower_bound[col]
                if df[col][i] > upper_bound[col]:
                    df[col][i] = upper_bound[col]
        for col in df.columns:
            return(df)


# In[25]:


df_new= mod_outlier(df)


# In[26]:


df_new.boxplot(figsize=(16,8))
plt.xticks(rotation=90)
plt.show()


# ### MISSING VALUE

# In[27]:


((df_X>UL) | (df_X<LL)).sum()


# In[28]:


df_X[((df_X>UL) |(df_X<LL))].sum().sum()


# In[29]:


df_X[((df_X>UL) |(df_X<LL))]=np.nan


# In[30]:


df_X.isnull().sum()


# In[31]:


df_X.isnull().sum().sum()


# In[32]:


df_X=df_X.drop(['Net_worth'],axis=1)


# In[33]:


df_X=df_X.drop(['Networth_Next_Year'],axis=1)


# In[34]:


df_X=df_X.drop(['Num'],axis=1)


# In[35]:


df_X.shape


# In[36]:


df_sub1=pd.concat([df_X,df_Y],axis=1)


# In[37]:


plt.figure(figsize=(20,20)) 
sns.heatmap(df_sub1.isnull(),cbar=False,cmap='coolwarm',yticklabels=True)
plt.show()


# In[38]:


cat=[]
num=[]
for i in df_new.columns:
    if df_new[i].dtype=="object":
        cat.append(i)
    else:
        num.append(i)
print(cat) 
print(num)


# In[39]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df_new = pd.DataFrame(imputer.fit_transform(df_new))
df_new.columns=num
#df_new =df_fra[cat]
df_new.head()


# In[40]:


df_new.isnull().any().any()


# In[41]:


df_new.isnull().sum().sort_values(ascending=False).head(20)


# In[ ]:





# In[42]:


df_sub1.isnull().sum(axis=1)


# In[43]:


df_sub1_temp=df_sub1[df_sub1.isnull().sum(axis=1)<=5]


# In[44]:


df_sub1_temp.shape


# In[45]:


df_sub1_temp['default'].value_counts()


# In[46]:


df_sub1.isnull().sum().sort_values(ascending=False)/df_sub1.index.size


# In[47]:


df_sub2=df_sub1.drop(['PE_on_BSE','Investments','Other_income','Contingent_liabilities','Deferred_tax_liability','Income_from_fincial_services','Equity_face_value','Change_in_stock'],axis=1)


# In[48]:


df_sub2.shape


# In[49]:


plt.figure(figsize=(35,10))
sns.boxplot(data=df_new)
plt.xlabel("Variables")
plt.xticks(rotation=90)
plt.ylabel("Density")
plt.title('Figure:Boxplot of few important features')


# In[ ]:





# In[ ]:





# In[50]:


sns.stripplot(df.Networth_Next_Year)


# In[51]:


sns.boxplot(df.Networth_Next_Year)


# In[52]:


sns.barplot(df.Networth_Next_Year)


# In[53]:


sns.stripplot(df.Total_assets )


# In[54]:


sns.boxplot(df.Total_assets )


# In[55]:


sns.barplot(df.Total_assets )


# In[56]:


sns.stripplot(df.Net_worth)


# In[57]:


sns.boxplot(df.Net_worth)


# In[58]:


sns.barplot(df.Net_worth)


# In[59]:


sns.barplot(df.PBDITA_as_perc_of_total_income )


# In[60]:


sns.boxplot(df.PBDITA_as_perc_of_total_income )


# In[61]:


sns.stripplot(df.PBDITA_as_perc_of_total_income )


# In[62]:


sns.boxplot(df.PAT_as_perc_of_total_income)


# In[63]:


sns.barplot(df.PAT_as_perc_of_total_income)


# In[64]:


sns.stripplot(df.PAT_as_perc_of_total_income)


# In[65]:


sns.boxplot(df.Cash_profit_as_perc_of_total_income )


# In[66]:


sns.barplot(df.Cash_profit_as_perc_of_total_income )


# In[67]:


sns.stripplot(df.Cash_profit_as_perc_of_total_income )


# In[68]:


fig=plt.subplots(figsize=(25,15))
sns.heatmap(df.corr(),annot=True)


# In[69]:


predictors=df_sub2.drop('default',axis=1)
response=df_sub2['default']


# In[70]:


from sklearn.preprocessing import StandardScaler


# In[71]:


scaler=StandardScaler()


# In[72]:


scaled_predictors=pd.DataFrame(scaler.fit_transform(predictors), columns=predictors.columns)


# In[73]:


df_sub3=pd.concat([scaled_predictors,response],axis=1)


# In[74]:


df_sub3.columns


# ### TRAIN TEST SPLIT

# In[75]:


from sklearn.model_selection import train_test_split
Train,Test= train_test_split(df_sub3,test_size=0.33,stratify=df_sub3.default)


# In[76]:


from sklearn.impute import KNNImputer


# In[77]:


Imputer= KNNImputer(n_neighbors=5)


# In[78]:


df_imputed_train= pd.DataFrame(Imputer.fit_transform(Train), columns=Train.columns)
df_imputed_test=pd.DataFrame(Imputer.transform(Test), columns=Test.columns)


# In[79]:


print(df_imputed_train.isnull().sum().sum())
print(df_imputed_test.isnull().sum().sum())


# In[80]:


X = df_new.drop(['default','Networth_Next_Year'], axis=1)
y = df_new['default']


# In[81]:


X.head()


# In[82]:


y.head()


# In[83]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)


# In[84]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler() 
X_train_scaled=ss.fit_transform(X_train)
X_test_scaled=ss.transform(X_test)


# In[85]:


X_train_scaled.shape


# In[86]:


X_train.shape


# In[ ]:





# In[ ]:





# In[87]:


plt.figure(figsize=(20,20))
cor_matrix=df_imputed_train.drop('default',axis=1).corr()
sns.heatmap(cor_matrix,cmap='plasma',vmin=-1,vmax=1)


# In[88]:


X_train=df_imputed_train.drop('default',axis=1)
y_train=df_imputed_train['default']
X_test=df_imputed_test.drop('default',axis=1)
y_test=df_imputed_test['default']


# ### LOGISTIC REGRESSION

# In[89]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[90]:


LogR=LogisticRegression()


# In[91]:


selector=RFE(estimator=LogR, n_features_to_select=14,step=1)


# In[92]:


selector = selector.fit(X_train, y_train)


# In[93]:


selector.n_features_


# In[94]:


selector.ranking_


# In[95]:


from sklearn.model_selection import GridSearchCV


# In[96]:


grid={'penalty':['l2','none'],
 'solver':['sag','lbfgs'],
 'tol':[0.0001,0.00001]}


# In[97]:


model = LogisticRegression(max_iter=10000,n_jobs=2)


# In[98]:


grid_search = GridSearchCV(estimator = model, param_grid = grid, cv = 3,n_jobs=-1,scoring='f1')


# In[99]:


grid_search.fit(X_train, y_train)


# In[100]:


print(grid_search.best_params_,'\n')
print(grid_search.best_estimator_)


# In[101]:


best_model = grid_search.best_estimator_


# In[102]:


ytrain_predict = best_model.predict(X_train)
ytest_predict = best_model.predict(X_test)


# In[103]:


ytest_predict_prob=best_model.predict_proba(X_test)
pd.DataFrame(ytest_predict_prob).head()


# ### Q 1.7 Validate the Model on Test Dataset and state the performance matrices. Also state interpretation from the model

# In[104]:


get_ipython().system('pip install --upgrade scikit-learn')


# In[105]:


from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.metrics.plot_confusion_matrix import plot_confusion_matrix


# In[ ]:


import sklearn
print(sklearn.__version__)


# In[ ]:


plot_confusion_matrix(best_model,X_train,y_train)
print(classification_report(y_train, ytrain_predict),'\n');


# In[ ]:


plot_confusion_matrix(best_model,X_test,y_test)
print(classification_report(y_test, ytest_predict),'\n');


# ### Q 1.8 Build a Random Forest Model on Train Dataset. Also showcase your model building approach.

# In[107]:


from sklearn.ensemble import RandomForestClassifier


# In[108]:


param_grid_rfcl = {
 'max_depth': [3,5,7],
 'min_samples_leaf': [5,10,15],
 'min_samples_split': [15,30,45],
 'n_estimators': [25,50]
}
rfcl = RandomForestClassifier(random_state=42)
grid_search_rfcl = GridSearchCV(estimator = rfcl, param_grid = param_grid_rfcl, cv =3)


# In[109]:


grid_search_rfcl.fit(X_train, y_train)


# In[110]:


grid_search_rfcl.best_params_


# In[111]:


best_grid_rfcl = grid_search_rfcl.best_estimator_


# In[ ]:


best_grid_rfcl


# In[ ]:


ytrain_predict_rfcl = best_grid_rfcl.predict(X_train)
ytest_predict_rfcl = best_grid_rfcl.predict(X_test)


# In[ ]:


ytest_predict_rfcl
ytest_predict_prob_rfcl=best_grid_rfcl.predict_proba(X_test)
ytest_predict_prob_rfcl
pd.DataFrame(ytest_predict_prob_rfcl).head()


# In[ ]:


print (pd.DataFrame (best_grid_rfcl.feature_importances_, 
 columns = ["Imp"], 
 index = X_test.columns).sort_values('Imp',ascending=False))


# In[ ]:


rfcl = RandomForestClassifier(max_depth=6, max_features=5, min_samples_leaf=8,
 min_samples_split=50, n_estimators=290, random_state=42)


# In[ ]:


rfcl = RandomForestClassifier(max_depth=6, max_features=5, min_samples_leaf=8,
 min_samples_split=50, n_estimators=290, random_state=42)


# In[ ]:


rf_clf = RandomForestClassifier(random_state=42)

# Create the feature selector object
selector_RF = SelectFromModel(estimator=rf_clf)

# Fit the feature selector to the data and transform the data
X_train_selected = selector_RF.fit_transform(X_train, y_train)


# In[ ]:


selector_RF = selector_RF.fit(X_train, y_train)


# In[ ]:


selector_RF.n_features_


# In[ ]:


selector_RF.ranking_


# In[ ]:


df = pd.DataFrame({'Feature':X.columns,'Rank': selector_RF.ranking_}) 
df[df['Rank'] == 1]


# In[ ]:


rf_test_fpr, rf_test_tpr,_=roc_curve(y_train,selector_RF.predict_proba(X_train)[:,1])
plt.plot(rf_test_fpr,rf_test_tpr,color='green')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (FPR)")
plt.title("Random Forest AUC-ROC for Train Data ")
rf_test_auc=roc_auc_score(y_train,selector_RF.predict_proba(X_train)[:,1])
print('Area under Curve is', rf_test_auc)
 
ytest_predict_rfcl
ytest_predict_prob_rfcl=best_grid_rfcl.predict_proba(X_test)
ytest_predict_prob_rfcl
pd.DataFrame(ytest_predict_prob_rfcl).head()


# In[ ]:




