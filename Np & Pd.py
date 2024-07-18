#!/usr/bin/env python
# coding: utf-8

# # NUMPY

# In[212]:


import sys 
import numpy as np


# In[8]:


A= np.array([
    [1,2,3], 
    [4,5,6]
])


# In[9]:


A.shape


# In[8]:


A.ndim


# In[12]:


a= np.array([0,1,2,3])


# In[13]:


a


# In[10]:


b= np.array([10,10,10,10])


# In[14]:


b


# In[15]:


a+b


# In[17]:


np.zeros(10)


# In[19]:


np.ones([2,2])


# In[23]:


np.identity(4)


# In[22]:


np.ones([4,4]) * 5


# In[24]:


#Create a numpy array, filled with 3 random integer values between 1 and 10.

np.random.randint(10, size=3)


# In[25]:


#Given the X python list convert it to an Y numpy array

X=[1,2,3,4]
print(X, type(X))

Y= np.array(X)
print(Y, type(Y))


# In[28]:


#Given the X numpy array, make a copy and store it on Y.

X=np.array([2,3,5])
print(X)

Y=np.copy(X)
print(Y)


# In[34]:


#odd numbers between 1 to 10

np.arange(1,11,2)

#in descending order
np.arange(1,11)[::-1]


# In[35]:


#Create a 3*3 numpy matrix, filled with values ranging from 0 to 8 

np.arange(9).reshape(3,3)


# In[40]:


#Show the memory size of the given Z numpy matrix

Z=np.zeros((10,10))
print('%d bytes'% (Z.size*Z.itemsize))


# In[41]:


#
X = [-5, -3, 0, 10, 40]

np.array(X, np.float)


# In[43]:


#make a mask showing negative elements
X = np.array([-1,2,0,-4,5,6,0,0,-9,10])

mask = X <= 0
X[mask]


# In[48]:


#DOT PRODUCT OF A & B
A= np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
B=np.array([
    [6,5],
    [4,3],
    [2,1]
])
A.dot(B)


# In[49]:


A@B


# In[50]:


B.T


# In[51]:


B.T @ A


# # PANDAS

# In[210]:


get_ipython().system('pip install pandas')


# In[211]:


import pandas as pd
import numpy as np


# # Pandas- Series

# In[59]:


S= pd.Series([1, 1.1, 1.2, 1.3, 1.4, 1.5])
S


# In[80]:


#Assigning random no.s to series
S=pd.Series(np.random.randn(5), name="begin")
S


# In[76]:


#renaming index
S.index=('Manvi','Mummy','Papa','Minnu','Dadi')
S


# In[72]:


S.describe()


# In[71]:


#absolute value of series

np.abs(S)


# In[108]:


#Conditional Selection(Boolean Arrays)
S=pd.Series([1,2.345,1.234,6.3456,33.456,94.32,12.345])
S


# In[109]:


S.index= ('A','B','C','D','E','F','G')
S


# In[110]:


S['G']


# In[111]:


#Conditional Selection(Boolean Arrays)
S>70


# In[112]:


#Conditional Selection(Boolean Arrays)
S[S>70]


# In[113]:


S.mean()


# In[114]:


S[S>S.mean()]


# In[115]:


S.std()


# In[119]:


#Operation&Methods
np.log(S)


# # Series- Exercises

# In[130]:


#Create a series and name it 'My Letters'
X = pd.Series(['A','B','C'], name='My Letters')
X


# In[133]:


#SHOW VALUES of give X series

X = pd.Series(['A','B','C'])
X.values


# In[16]:


#Print First and Last Element

X= pd.Series(['A','B','C'])
X.iloc[[0,-1]]


# In[ ]:


#SERIES MANIPULATION
#Convert to Float

X= pd.Series(['A','B','C','D','E'])
pd.Series(X, dtype=np.float)


# In[20]:


#Order (sort) the given pandas Series

X = pd.Series([2,4,1,3,8,0])
X=X.sort_values()
X


# In[28]:


#boolean arrays (also called masks)

#Given the X pandas Series, make a mask showing negative elements
X= pd.Series([-1,2,0,-4,5,6,0,0,-9,10])
mask= X<=0
mask


# In[42]:


#Given the X pandas Series, make a mask to GET ALL negative elements
X= pd.Series([-1,2,0,-4,5,6,0,0,-9,10])
mask= X<=0
X[mask]


# In[45]:


# get numbers higher than the elements mean
X=([1,2,3,5,1,7,8,5,9])
mask= X> X.mean()
X[mask]


# In[7]:


#Get Numbers equal to 2 or 10
X = pd.Series([-1,2,0,-4,5,6,0,0,-9,10])

mask = (X == 2) | (X == 10)
X[mask]


# In[6]:


#Given the X pandas Series, show the max value of its elements
X = pd.Series([1,2,0,4,5,6,0,0,9,10])
X.max()


# # Pandas- Data Frames

# In[10]:


#Table

df = pd.DataFrame({
    'Population': [35.467, 63.951, 80.94 , 60.665, 127.061, 64.511, 318.523],
    'GDP': [
        1785387,
        2833687,
        3874437,
        2167744,
        4602367,
        2950039,
        17348075
    ],
    'Surface Area': [
        9984670,
        640679,
        357114,
        301336,
        377930,
        242495,
        9525067
    ],
    'HDI': [
        0.913,
        0.888,
        0.916,
        0.873,
        0.891,
        0.907,
        0.915
    ],
    'Continent': [
        'America',
        'Europe',
        'Europe',
        'Europe',
        'Asia',
        'Europe',
        'America'
    ]
}, columns=['Population', 'GDP', 'Surface Area', 'HDI', 'Continent'])


# In[11]:


df


# In[14]:


df.columns


# In[15]:


df.size


# In[23]:


df.info()


# In[20]:


df.describe()


# In[26]:


df.shape


# In[36]:


df.dtypes


# In[34]:


df.dtypes.value_counts() 


# # DF- Indexing,selection and slicing

# In[39]:


df


# In[43]:


df.index = [
    'Canada',
    'France',
    'Germany',
    'Italy',
    'Japan',
    'United Kingdom',
    'United States',
]


# In[55]:


df


# In[45]:


#FOR ROWS
#loc attribute shows the rows by index 
df.loc['Canada']


# In[47]:


#iloc attribute displays rows by sequential position
df.iloc[-2]


# In[50]:


#FOR COLUMNS
#to select a specific column
df['Population']


# In[51]:


#select multiple columns
df[['Population','GDP']]


# In[57]:


#Slicing 
#Display data from Canada to Italy
df.loc['Canada':'Italy']


# In[58]:


#Display from row 1 to 4
df.iloc[0:4]


# In[61]:


#Select specific rows and columns
df.loc['Canada':'France',['GDP','HDI']]


# In[66]:


df.iloc[1:3,[1,3]]


# # DF- Conditional selection(boolean arrays)
# 

# In[67]:


df['Population']>70


# In[68]:


df.loc[df['Population']>70]


# In[72]:


df.loc[df['Population']>70,['Population','GDP']]


# # DF- Dropping Stuff

# In[73]:


df.drop('Canada')


# In[79]:


df.drop(['Canada', 'United States','Japan','Italy'])


# In[80]:


df.drop(columns=['Population','GDP'])


# In[90]:


df.drop(['Italy','Canada'],axis=0)


# # DF- Modifying 

# In[113]:


#Adding New Column
langs=pd.Series(['English','English', 'Japanese','English','Spanish'y,'French','German'])
name='Language'
langs


# In[114]:


df['Language']='English'
df


# In[117]:


#Renaming Columns
df.rename(columns={'GDP':'Gross Domestic Product',
                   'HDI':'Human Development Index'},
          index={'United States':'USA',
                 'United Kingdom':'UK',
                 'Argentina':'AR',
                'Canada':'CN'})


# In[118]:


df


# In[127]:


#Creating columns from other columns 
df['GDP per Capita']= df['GDP']/df['Population']
df


# # Data Frames- Exercises

# In[207]:


get_ipython().system('pip install matplotlib')


# In[209]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[132]:


#DataFrame creation
#Create an empty pandas DataFrame
pd.DataFrame(data=[None],
             index=[None],
             columns=[None])


# In[139]:


marvel_data = [
    ['Spider-Man', 'male', 1962],
    ['Captain America', 'male', 1941],
    ['Wolverine', 'male', 1974],
    ['Iron Man', 'male', 1963],
    ['Thor', 'male', 1963],
    ['Thing', 'male', 1961],
    ['Mister Fantastic', 'male', 1961],
    ['Hulk', 'male', 1962],
    ['Beast', 'male', 1963],
    ['Invisible Woman', 'female', 1961],
    ['Storm', 'female', 1975],
    ['Namor', 'male', 1939],
    ['Hawkeye', 'male', 1964],
    ['Daredevil', 'male', 1964],
    ['Doctor Strange', 'male', 1963],
    ['Hank Pym', 'male', 1962],
    ['Scarlet Witch', 'female', 1964],
    ['Wasp', 'female', 1963],
    ['Black Widow', 'female', 1964],
    ['Vision', 'male', 1968]
] 
marvel_df


# In[138]:


marvel_df=pd.DataFrame(data=marvel_data)
marvel_df


# In[140]:


#Add column names to the marvel_df
column=['name','sex','first appearance']
marvel_df.columns=column
marvel_df


# In[142]:


#Add index names to the marvel_df(use the character name as index)
marvel_df.index=marvel_df['name']
marvel_df


# In[169]:


marvel_df=marvel_df.drop(['Vision','Wasp'])
marvel_df


# In[219]:


#Show the first 5 elements on marvel_df
marvel_df[1:6]


# In[205]:


#Show the last 5 elements
marvel_df[-5:]


# In[204]:


#Show just the sex of the first 5 elements
marvel_df.iloc[:5,].sex.to_frame()


# In[199]:


marvel_df.iloc[[0, -1],]


# In[203]:


marvel_df['years_since'] = 2018 - marvel_df['first_appearance']

marvel_df


# # Sharing external data & Plotting

# In[227]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[228]:


pd.read_csv


# In[232]:


df = pd.read_csv('Downloads/btc-market-price.csv')


# In[235]:


df.head()


# In[238]:


df = pd.read_csv('Downloads/btc-market-price.csv' , header=None)
df.head()


# In[240]:


df.columns=['Timestamp','Price']
df.head()


# In[242]:


df.shape


# In[252]:


pd.to_datetime(df['Timestamp']).head()
df['Timestamp']=pd.to_datetime(df['Timestamp'])
df.head()


# In[254]:


df.set_index('Timestamp', inplace=True)
df.head()


# In[258]:


df.loc['2017-04-06']


# In[261]:


#long method
df = pd.read_csv('Downloads/btc-market-price.csv', header=None)
df.columns = ['Timestamp', 'Price']
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)
df.head()


# In[268]:


#shorter way

df= pd.read_csv(
    'Downloads/btc-market-price.csv',
    header=None,
    names=['Timestamp','Price'],
    index_col=0,
    parse_dates=True)
df.head()


# # Basic Plotting

# In[270]:


df.plot()


# In[272]:


plt.plot(df.index, df['Price'])


# In[280]:


plt.plot(x,x**2)


# In[281]:


plt.plot(x, x ** 2)
plt.plot(x, -1 * (x ** 2))


# In[288]:


plt.figure(figsize=(4, 4))
plt.plot(x, x ** 2)
plt.plot(x, -1 * (x ** 2))

plt.title('My Nice Plot')

