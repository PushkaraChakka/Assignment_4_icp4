#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pandas
import pandas as pd
df=pd.read_csv("data.csv")#to read csv file
df.describe()#to describe basic ststical description from data


# In[2]:


show_null=df.isnull().sum()#display null values
print(show_null)#print the null values


# In[3]:


df.fillna(df.mean(),inplace=True)#replace null values with mean value
print(df)#print after replacing with mean


# In[4]:


df=df[["Duration","Pulse","Maxpulse","Calories"]]
aggregate={"Duration":["max","min","count","mean"],
          "Pulse":["max","min","count","mean"],
          "Maxpulse":["max","min","count","mean"],
          "Calories":["max","min","count","mean"]}# to find max,min,count,mean of all the columns
aggregate_df=df.agg(aggregate)#function to aggregate
print(aggregate_df)#print the aggregate


# In[5]:


calories_in_range=(df["Calories"]>=500) & (df["Calories"]<=1000)#defining range of values to be displayed
filters_result=df[calories_in_range]#adding the defined range to new variable
print(filters_result)#printing the new result


# In[6]:


calories_pulse_filter=(df["Calories"]>500)&(df["Pulse"]<100)#defining range
filters_result=df[calories_pulse_filter]#adding the result to new variable
print(filters_result)#printing the result


# In[7]:


df_modified=df.drop(columns=["Maxpulse"])#displaying every column except Maxpulse
print(df_modified)#printing the rersult


# In[8]:


del df["Maxpulse"]#command to delete entire row
print(df)


# In[9]:


df['Calories'] = df['Calories'].fillna(0).astype(int)#converting to int data type
print(df)


# In[16]:


import matplotlib.pyplot as plt
df.plot(kind='scatter', x='Duration', y='Calories', figsize=(6,3))
plt.show()


# In[19]:


import pandas as pd 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


salariesData = pd.read_csv('Salary_Data (2).csv') #importing data from the CSV file
df.describe()


# In[20]:


#splitting the data in to training and testing
X = salariesData.iloc[:, :-1].values 
Y= salariesData.iloc[:, 1].values


#splitting 1/3 of the data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)


# In[21]:


# Fitting Simple Linear Regression to the training set
reg = LinearRegression()
reg.fit(X_train, Y_train)


# Predicting the Test set result 
pred = reg.predict(X_test)


# Calculating the Mean_squared_error
mse = mean_squared_error(Y_test, pred)


#Visualising the Training set results and Test set results
plt.scatter(X_train, Y_train, color = 'blue')
plt.scatter(X_test, Y_test, color = 'red')
plt.title('Salary Data')
plt.xlabel('Experience (Years)')
plt.ylabel('Salary')
plt.show()


# In[ ]:




