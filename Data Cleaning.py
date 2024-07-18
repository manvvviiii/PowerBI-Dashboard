#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning

# In[3]:


import pandas as pd
import numpy as np


# In[3]:


pd.isnull(np.nan)


# In[5]:


pd.isna(np.nan)


# In[8]:


pd.isnull(None)


# In[9]:


pd.notnull(3)


# In[12]:


pd.isnull(pd.Series([1,np.nan,7]))


# # Pandas operations with missing values

# In[13]:


pd.Series([1,5,np.nan]).sum()


# In[4]:


pd.Series([2,np.nan,3,6,2,np.nan]).count()


# # Filtering Missing Data

# In[5]:


s=pd.Series([1,4,np.nan,7,np.nan,2,4,8,np.nan])


# In[20]:


s.notnull()


# In[25]:


s[s.isnull()]


# In[15]:


#Dropping Null Values
s.dropna()


# In[36]:


df=pd.DataFrame({
    'Column A':[4,6,np.nan,1],
    'Column B':[4,3,9,2],
    'Column C':[5,np.nan,2,np.nan],
    'Column D':[8,3,4,1]
})
df


# In[28]:


df.isnull()


# In[30]:


df.isnull().sum()


# In[40]:


df.dropna()


# In[41]:


df.dropna(how='all')


# In[42]:


df.dropna(how='any')


# In[49]:


df.dropna(thresh=2,axis=0)


# # Filling Null Values

# In[51]:


s.fillna(0)


# In[64]:


s.fillna(s.mean())


# In[66]:


#FILLING NULLSWITH CONTIGUOUS(Close) VALUES
#Forward FIll
s.fillna(method='ffill')


# In[67]:


#Backward Fill
s.fillna(method='bfill')


# # Checking if there are NAs

# In[70]:


s.dropna().count()


# In[74]:


missingvalues=len(s.dropna())!=len(s)
missingvalues


# In[72]:


len(s)


# In[75]:


missingvalues=s.count()!=len(s)
missingvalues


# In[78]:


s.isnull().values


# # Duplicates

# In[80]:


ambassadors = pd.Series([
    'France',
    'United Kingdom',
    'United Kingdom',
    'Italy',
    'Germany',
    'Germany',
    'Germany',
], index=[
    'Gérard Araud',
    'Kim Darroch',
    'Peter Westmacott',
    'Armando Varricchio',
    'Peter Wittig',
    'Peter Ammon',
    'Klaus Scharioth '
])
ambassadors


# In[81]:


ambassadors.duplicated()


# In[82]:


ambassadors.duplicated(keep='last')


# In[84]:


ambassadors.drop_duplicates()


# In[86]:


ambassadors.duplicated(keep='first')


# # Duplicates in DF

# In[87]:


players = pd.DataFrame({
    'Name': [
        'Kobe Bryant',
        'LeBron James',
        'Kobe Bryant',
        'Carmelo Anthony',
        'Kobe Bryant',
    ],
    'Pos': [
        'SG',
        'SF',
        'SG',
        'SF',
        'SF'
    ]
})
players


# In[91]:


players.duplicated()


# In[97]:


players.duplicated(subset=('Name'))


# # Text Handling

# # Splitting Columns

# In[172]:


df = pd.DataFrame({
    'Data': [
        '1987_M_US _1',
        '1990?_M_UK_1',
        '1992_F_US_2',
        '1970?_M_   IT_1',
        '1985_F_I  T_2'
]})
df


# In[173]:


df = df['Data'].str.split('_', expand=True)
df


# In[174]:


df.columns=['Year','Age','Country','No Children']


# In[175]:


df


# In[176]:


df['Year'].str.contains('\?')


# In[177]:


df['Country'].str.contains('U')


# In[178]:


df['Country'].str.strip()


# In[188]:


df['Country'].str.replace(' ','')


# In[193]:


df['Year'].str.replace('?','')

