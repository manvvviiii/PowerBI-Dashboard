#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import random
get_ipython().run_line_magic('matplotlib', 'inline')


# # Global API

# In[24]:


plt.figure(figsize=(4, 3))

plt.title('My First Plot')

plt.plot(x, x ** 2)
plt.plot(x, -1 * (x ** 2))


# In[28]:


plt.figure(figsize=(11,4))
plt.title("Plot 1")

plt.subplot(1,2,1) #Rows,columns,Panel Selected
plt.plot(x,x**2)
plt.plot([0,0,0],[-10,0,100])
plt.legend(['X^2', 'Vertical Line'])
plt.xlabel('X')
plt.ylabel('X squared')

plt.subplot(1,2,2)
plt.plot(x,-1*(x**2))
plt.plot([-10,0,10],[-50,-50,-50])
plt.legend('-X^2','Horizontal Line')
plt.xlabel('X')
plt.ylabel('X Squared')


# In[22]:


plt.figure


# # OOP API

# In[74]:


fig,axes=plt.subplots(figsize=(11,4))


# In[78]:


axes.plot(x, (x ** 2), marker='o',color='red',label='X^2')
axes.plot(x, -1 * (x ** 2),linestyle='dashed',linewidth=2,label='-X^2')
axes.set_xlabel('X')
axes.set_xlabel('X squared')
axes.set_title('OOP API')
axes.legend()
fig


# In[87]:


print('Markers: {}'.format([m for m in plt.Line2D.markers]))


# # Other types of Plots

# # Figures and Subfigures

# In[88]:


new = plt.subplots()

fig, ax = new

ax.plot([1,2,3], [1,2,3])

new


# In[95]:


x=np.arange(-10,10)
new=plt.subplots(nrows=2,ncols=2,figsize=(6,6))
fig,((ax1,ax2),(ax3,ax4))=new
new


# In[102]:


new=plt.subplots(nrows=2,ncols=2,figsize=(6,6))
fig,((ax1,ax2),(ax3,ax4))=new
ax1.plot(np.random.randn(50),c='Blue')
ax2.plot(np.random.randn(10),c='red')
ax3.plot(np.random.randn(20),c='Green')
ax4.plot(np.random.randn(100),c='yellow')
new


# # subplot2 Grid command

# In[119]:


plt.figure(figsize=(14, 6))

ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
ax2 = plt.subplot2grid((3,3), (1,0), colspan=2)
ax3 = plt.subplot2grid((3,3), (1,2))
ax4 = plt.subplot2grid((3,3), (2,0))
ax5 = plt.subplot2grid((3,3), (2,1))


ax1.plot(np.random.randn(100),c='Blue')
ax2.plot(np.random.randn(10),c='red')
ax3.plot(np.random.randn(20),c='Green')
ax4.plot(np.random.randn(80),c='yellow')
ax5.plot(np.random.randn(50),c='pink')


# # Scatter Plot

# In[185]:


x=np.random.randn(N)
y=np.random.randn(N)
plt.scatter(x,y,alpha=0.8,marker='^')
plt.show()


# In[201]:


x=[20,10,33,11,39,38,49,50]
y=[40.2,19,30,93,22,18,72,66]
colors=['r','g','b','y','b','black','pink','g']
area=[567,254,473,282,444,664,265,293]
plt.scatter(x,y,c=colors,alpha=0.8,s=area,marker='*',edgecolor='black')
plt.show()


# In[207]:


#Using cmap and colorbar 
plt.figure(figsize=(8, 4))

plt.scatter(x, y,s=area, alpha=0.9,c=colors, cmap='Pastel1')
plt.colorbar()

plt.show()


# In[ ]:





# # Histogram 

# In[63]:


num=np.random.randn(50)
plt.hist(num,color='pink',bins=10,alpha=0.5,edgecolor='black',cumulative=-1,align='left', rwidth=0.8,label='python')
plt.xlim(xmin=-6,xmax=6)
plt.axvline(1.2,color='black',alpha=0.6,linestyle='dashed',label='value=1.2')
plt.legend()
plt.grid()
plt.show()


# # KDE(Kernel Density Estimation)

# In[64]:


get_ipython().system('pip install scipy')


# In[65]:


from scipy import stats


# In[80]:


df=pd.DataFrame({'Country':['India','USA','Africa','Japan','UK'],
                 'Population':[35.467, 63.951, 80.94 , 60.665, 127.061],
                 'State':['Gujarat','Texas','Kenya','Tokyo','London']
                })
df


# In[84]:


df.plot.kde(color='orange')


# # Bar Plot

# In[139]:


x=['Hindi','English','Math','Science','SS']
y=[85,72,95,78,60]
z=[75,80,81,85,55]
BarWidth=0.2
plt.figure(figsize=(9,4))
plt.bar(x,y,color='blue',edgecolor='Black',label='subjects')
plt.bar(x,z,color='red',alpha=0.4,edgecolor='black',lw=0.5,label='Previous Sem Marks')
plt.xlabel("Subjects")
plt.ylabel("Marks")
plt.axhline(65,color='red',linestyle='dashed',label='Average',alpha=0.7)
plt.title("Student Marks")
plt.grid()
plt.legend(fontsize=11)
plt.show()


# In[163]:


x=['Hindi','English','Math','Science','SS']
y=[85,72,95,78,60]
z=[75,80,81,85,55]

BarWidth=0.3
p=np.arange(len(x))
p1=[j+BarWidth for j in p]

plt.bar(p,y,color='yellow',edgecolor='Black',label='subjects',width=0.3,alpha=0.7)
plt.bar(p1,z,color='pink',edgecolor='black',lw=0.5,label='Previous Sem Marks',width=0.3)

plt.xlabel("Subjects")
plt.ylabel("Marks")
plt.axhline(65,color='red',linestyle='dashed',label='Average',alpha=0.7)
plt.title("Student Marks")

plt.xticks(p+BarWidth,x)
plt.grid()
plt.legend()
plt.show()


# In[ ]:




