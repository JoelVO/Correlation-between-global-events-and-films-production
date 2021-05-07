The imported libraries were

```python
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import collections
import seaborn as sns
import tensorflow as tf
import scipy
```

I started this project by condensing the information I needed from the DGELT Project database. Using pandas, I explored all the databases (because of the amount of information, GDELT divided it into several datasets) calculating which amount of the news published in some given year where belonged to some given category of the CAMEO guidelines.

For example, for the dataset corresponding to the news published on april, 2013, the code was 

````python
for i in range(1,31): #one csv per day of that month
    s='/Users/52553/Downloads/201304'
    if i<10:
        j='0'+str(i)
    else:
        j=i
    pelis=pd.read_csv(s+str(j)+'.export.csv', sep='\t') #reading the tsv as a csv
    a=collections.Counter(pelis.iloc[:,28])
    b=[]
    for j in range(1,21):
        b.append(a[j])
    b=b/np.sum(b)
    #print(i,b)
    b=np.concatenate((['201304'+str(i)],b)) #doing the calculations
    A.append(b) #storing data
````
where A was a list which would become a dataframe at the end of the preprocessing part.

For the preprocessing of the movies, if was interested in knowing, with some given year, the proportion of movies released on that year which belonged to a given genre. The code for this was 

````python
col=np.unique(np.array(df.genres)) # array with all the possible movie genres
A=np.zeros((len(col),2021-1979)) #matrix of zeros

#counting the disired proportions
for j in range(len(col)): #for each movie genre
    for i in range(2021-1979):  #for each year between 1979 and 2021
        A[j][i]=((df[(df[col[j]]==1) & (df['fechas']==(1979+i))].shape[0])/df[df['fechas']==(1979+i)].shape[0]) #storing the desired proportion
        
````

As a result of this last block of code, I was able to make the following graphs where the genre "N" corresponds for none. This was made so I could have a grasp on the behaviour of the data I was working with.

![](/images/movies_vs_time_1.png)
![](/images/movies_vs_time_2.png)
![](/images/movies_vs_time_3.png)

As we can see, some genres have became more popular through time, such as horror of documentary.

Looking for correlation between the data from the movies and the news, I calculated the correlation matrix with the following code

````python
B=pd.read_csv('/Users/52553/OneDrive/Escritorio/pelis_tiempo.csv') #reading the processed data from the movies
D=pd.read_csv('/Users/52553/OneDrive/Escritorio/eventos_tiempo.csv') #reading the processed data from the news

result = pd.concat([B, D], axis=1, sort=False) #making one dataframe from the previous ones
corr_mat=result.corr() #calculating the correlation matrix

plt.figure(figsize = (35,30)) #setting a figure
sns.heatmap(corr_mat,annot=True) #transforming the correlation matrix into a heatmap
plt.show() #showing the constructed image

````
 This code gave as output the next image, where the names on the axis are the movie genres and the possible classifications of the CAMEO guidelines repressented by numbers from 0 to 19.
 
 ![](/images/correlation_matrix.png)

The mathematical expression used for this calculation is the one from statistics and I think it is worthy to speak about its interpretation. When calculating the correlation between two series of data, one try to adjust a lineal function between both series taking one of then as a independent variable and the other one as the dependent variable. The sign of the result tells us if the relation is positive (an increment on the independent variable generates an increment on the dependent variable) or if it is negative (an increment on the independent variable produces a decrement on the dependent variable); on the other side, the absolute value of the result tells us how big was the effect found between the variables: if the absolute value is between 0 and 0.3, the effect is little; for absolute values between 0.3 and 0.5, there is evidence of a medium effect; and when the absolute value is bigger than 0.5, one may say the effect is big.

With this in mind, I may say I found several correlations between the type of news and the relevance on a movie genre, so I may affirm that the results of the model I am about to present are not just a numerical coincidences or an ad hoc function made using approximations in some space where the discrete topology was induced.

To build the DNN, I used the following code.

````python

df=pd.read_csv('/Users/52553/OneDrive/Escritorio/eventos_tiempo.csv') #preocessed data from the news
df1=pd.read_csv('/Users/52553/OneDrive/Escritorio/pelis_tiempo.csv') #processed data from the movies

Y=df1.to_numpy()
X=df.to_numpy()

X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

model = tf.keras.Sequential() #creating a sequential model
model.add(tf.keras.Input(shape=(20,)))  #the input were the data from the news, therefore I needed 20 input units, one for each possible classification of the news
model.add(tf.keras.layers.Dense(20, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='tanh')) #I experimented with several activation functions and these proved to be the best
model.add(tf.keras.layers.Dense(len(Y[0]),activation='softmax')) #the output will be the data from the movies
#since the output data is a vector in which each entry is the proportion of movies of the corresponding genre made that year, the velues are between 0 and 1
#even more, again by the meaning of the entries, the sum of all the entries from the vector is equal to 1
#then, I can think this vector as a discrete probability function and therefore think of this problem as a model for classification
          
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['CategoricalCrossentropy'])  

````

