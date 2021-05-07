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
A=np.zeros((len(col),2021-1979)) #matrix

#counting the disired proportions
for j in range(len(col)): #for each movie genre
    for i in range(2021-1979):  #for each year between 1979 and 2021
        A[j][i]=((df[(df[col[j]]==1) & (df['fechas']==(1979+i))].shape[0])/df[df['fechas']==(1979+i)].shape[0]) #storing the desired proportion
        
````

As a result of this last block of code, I was able to make the following graphs where the genre "N" corresponds for none.

![](\images\movies_vs_time_1.png)
