#!/usr/bin/env python
# coding: utf-8

# # Reading Data with Python

# In[10]:


with open('Downloads/btc-market-price.csv') as fp:
   print(fp)


# In[46]:


with open('Downloads/btc-market-price.csv') as file:
    for index,line in enumerate(file.readlines()):
        if (index<10):
           print(index,line)


# # Reading data with pandas

# # Reading csv file

# In[9]:


import pandas as pd

first='Downloads/btc-market-price.csv'
pd.read_csv(first)


# In[10]:


pd.read_csv(first).head()


# In[18]:


df=pd.read_csv('Downloads/btc-market-price.csv', 
               header=None, 
               names=['Timestamp','Price'],
               index_col=[0],
               parse_dates=[0],
               dtype={'Price':'float'}              
              )
df.head()


# In[37]:


df= pd.read_csv('Downloads/exam_review.csv', 
            sep='>',
           thousands=',')
df


# In[66]:


df=pd.read_csv('Downloads/exam_review.csv',
            sep='>',
           thousands=',',
           skiprows=2,
           skip_blank_lines=False)
df


# In[65]:


df=pd.read_csv('Downloads/exam_review.csv',
            sep='>',
           thousands=',',
           usecols=['first_name','last_name','age'])
df


# In[68]:


#CSV String
df.to_csv()


# # Reading data from relational databases

# In[75]:


get_ipython().system('pip install sqlalchemy')
import sqlite3 
import pandas as pd


# In[119]:


conn=sqlite3.connect('Downloads/chinook.db')


# In[106]:


cur =first.cursor()
cur.execute('SELECT * FROM employees LIMIT 5;')


# In[127]:


result =cur.fetchall()

df=pd.DataFrame(result)
df
# In[120]:


df=pd.read_sql('SELECT * from employees;', conn)
df


# In[112]:


df['ReportsTo'].mean()


# In[114]:


df['ReportsTo'].isna()


# In[125]:


pd.read_sql_query('SELECT * from employees;',conn, index_col='EmployeeId',)

