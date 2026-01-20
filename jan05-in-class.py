# %%

import pandas as pd
import numpy as np

#%%

df = pd.read_csv('data/airbnb_NYC.csv', encoding='latin1')
df.head(5)

# %%

df.dtypes
# %%

x = df['Price'].str.replace(',','')
pd.to_numeric(x, errors='coerce').hist(bins=100)