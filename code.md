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
 This code gave as output the next image.
