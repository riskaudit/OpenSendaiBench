# %%
import pandas as pd

# building type per country
data = pd.read_csv('bldgtype_country.csv')
labels = {}
for i in range(data.shape[0]):
    labels[data['Country'][i]] = list(data.columns[(data == 1).iloc[i]])

# signals per country
data1 = pd.read_csv('signals_country.csv')
signals = {}
for i in range(data1.shape[0]):
    signals[data1['Country'][i]] = list(data1.columns[(data1 == 1).iloc[i]])

# number of tiles per country
data2 = pd.read_csv('ntiles_country.csv')
ntiles = {}
for i in range(data1.shape[0]):
    ntiles[data1['Country'][i]] = data2.iloc[i]['ntiles']
# %%
