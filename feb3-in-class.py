# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import seaborn as sns

def minmax(x):
    xnp = x.to_numpy()
    u = (xnp-min(xnp)) / (max(xnp)-min(xnp))
    return u

# %%

df = pd.read_csv('../data/cars_env.csv')
df.head()

df['class'] = df['EPA class']
df['class'] = df['class'].replace(['MIDSIZE CARS','COMPACT CARS','SUBCOMPACT CARS','TWO SEATERS','LARGE CARS'],'car')
df['class'] = df['class'].replace(['SMALL STATION WAGONS','MIDSIZE STATION WAGONS'],'station wagon')
df['class'] = df['class'].replace(['STANDARD PICKUP TRUCKS','SMALL PICKUP TRUCKS'],'truck')
df['class'] = df['class'].replace(['VANS','MINIVAN'],'van')

y = df['class'] # Set out outcome/target
ctrl_list = ['baseline mpg', 'baseline price'] # List of control variables
x = df.loc[:, ctrl_list] # Set our covariates/features
u = x.apply(minmax) # Scale our variables

# %%

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

u_train, u_test, y_train, y_test = train_test_split(u,y,test_size=.2,random_state=100) 

kgrid = [(2*k+1) for k in range(100)]
accuracy = []
for k in kgrid:
    model = KNeighborsClassifier(n_neighbors=k)
    model = model.fit(u_train, y_train)
    # y_hat = model.predict(u_test)
    # accuracy.append(model.score(y_test, y_hat))

#sns.lineplot(x=kgrid, y=accuracy)

# %%

from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=15)
model = model.fit(u, y)
# y_hat = model.predict(u)

#sns.scatterplot(x=y, y=y_hat, alpha=0.1)