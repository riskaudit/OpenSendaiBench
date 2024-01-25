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
# %%
