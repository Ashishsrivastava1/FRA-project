#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color= sns.color_palette()
import sklearn.metrics as metrics
from scipy.stats import norm

import warnings
warnings.filterwarnings("ignore")


# In[2]:


df1=pd.read_csv('C:/Users/ASHISH SRIVASTAVA/Downloads/Company(FRA).csv')


# In[3]:


df1.head()


# In[4]:


df1.columns = df1.columns.str.replace(' ', '_').str.replace('.', '')


# In[5]:


df1.head()


# In[6]:


print('The number of rows (observations) is',df1.shape[0],'\n''The number of columns (variables) is',df1.shape[1])


# In[7]:


df1.info()


# In[8]:


df1.duplicated().sum()


# In[9]:


df1.describe()


# ### 2.1 Draw Stock Price Graph(Stock Price vs Time) for any 2 given stocks with inference

# In[12]:


df1['Nums'] = [pd.to_datetime(d) for d in df1['Num']]


# In[13]:


plt.figure(figsize = (10, 8))
plt.scatter(df1['Nums'], df1['Networth_Next_Year'], edgecolors='b', color = 'lightblue')
plt.xlabel('Year')
plt.ylabel('Networth_Next_Year')
plt.title('Networth_Next_Year over the years')
plt.show()


# In[14]:


plt.figure(figsize = (10, 8))
plt.scatter(df1['Nums'], df1['PBDITA'], edgecolors='r', color = 'pink')
plt.xlabel('Year')
plt.ylabel('PBDITA')
plt.title('PBDITA over the years')
plt.show()


# ###  2.2 Calculate Returns for all stocks with inference.

# In[ ]:





# In[ ]:





# In[15]:


df1 = np.log(df1.drop(['Num','Nums'],axis=1)).diff(axis = 0, periods = 1) 


# In[16]:


df1.shape


# In[17]:


df1.head()


# ### We now look at Means & Standard Deviations of these returns

# In[18]:


stock_means = df1.mean(axis = 0)
stock_means


# In[19]:


stock_sd = df1.std(axis = 0)
stock_sd


# In[20]:


df = pd.DataFrame({'Average':stock_means, 'Volatility': stock_sd})
df


# In[21]:


plot = sns.scatterplot(df['Volatility'], df['Average'])
plot.axvline(x=0.020257,linestyle='--', color = "red")
plot.axhline(y=0.000683,linestyle='--', color = "red")
plt.show()


# In[30]:


df.loc['Total_assets'][0]


# In[31]:


df[df['Average'] > df.loc['Total_assets'][0]].sort_values(by = 'Volatility')


# In[32]:


df.columns


# In[35]:


import statsmodels.formula.api as SM


# In[41]:


f_1 = 'Default ~Average+Volatility '


# In[42]:


model_1 = SM.logit(formula = f_1, data=Default).fit()


# In[43]:


model_1 = SM.logit(formula = f_1, data=Default).fit()


# In[ ]:




